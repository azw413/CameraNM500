#include <iostream>
#include <strstream>
#include <time.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <nmengine.h>

// Classes
#define CLASS_NOTHING        1
#define CLASS_PEOPLE         2
#define CLASS_CAR_UPHILL     3
#define CLASS_CAR_DOWNHILL   4

#define NUM_CLASSES 4


void tc_network_type(nm_device *device);
void tc_get_context(nm_device *device);
void tc_set_context(nm_device *device, uint8_t context, uint8_t norm, uint16_t minif, uint16_t maxif);
void tc_learn(nm_device *device, uint8_t* data, uint16_t size, uint16_t category);
void tc_classify(nm_device *device, uint8_t data, uint16_t size, uint8_t k);
void tc_read_neuron(nm_device *device, uint32_t nid, uint16_t size);
void tc_read_neurons(nm_device *device, uint16_t size);
void tc_write_neurons(nm_device *device, uint16_t size);
void tc_model_analysis(nm_device *device);
void tc_learn_batch(nm_device *device, uint16_t size, uint32_t count);
void tc_clusterize(nm_device *device);
void tc_save_model(nm_device *device, const char *path);
void tc_load_model(nm_device *device, const char *path);
void train_images(nm_device *device, char * folder, uint16_t category);

using namespace cv;

int main() {
    int totals[NUM_CLASSES+1];

    cv::VideoCapture cap;
    cv::Mat frame;
    const std::string videoStreamAddress = "rtsp://admin:123456@192.168.1.11:8554/profile0?tcp";

    //open the video stream and make sure it's opened
    if(!cap.open(videoStreamAddress))
    {

        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // Open the NM500
    uint16_t r;
    nm_device ds[10];
    uint8_t detect_count = 10;
    r = nm_get_devices(ds, &detect_count);
    if (r != NM_SUCCESS) {
        if (r == NM_ERROR_DEVICE_NOT_FOUND) {
            printf("[devices] Not found \n");
        }
        else {
            printf("[devices] Failed to get device list %d\n", r);
        }
        return 0;
    }

    if (detect_count < 1) {
        printf("[devices] There is no detected device\n");
        return 0;
    }

    printf("[devices] %d\t detected\n", detect_count);
    for (int i = 0; i < detect_count; i++) {
        printf("ID: %d\t TYPE: %d\t PID: %d\t VID: %d\n", i, ds[i].type, ds[i].vid, ds[i].pid);
    }

    nm_device *target = &ds[0];

    r = nm_connect(target);
    if (r != NM_SUCCESS) {
        printf("[init] Failed initialize NM500, Error: %d, or Not supported device\n", r);
        return 0;
    }

    // clear down device
    r = nm_reset(target);
    if (r != NM_SUCCESS) {
        printf("[init] Failed initialize NM500, Error: %d, or Not supported device\n", r);
        return 0;
    }


    // Learn the training images
    nm_context context;
    context.context = 0;
    context.maxif = 10000;
    context.minif = 128;
    context.norm = L1;

    nm_set_network_type(target, RBF);
    //nm_set_context(target, &context);

    // Initialise the network with training images
    train_images(target, "/home/awhaley/CLionProjects/CameraNM500/images/Nothing", CLASS_NOTHING);
    train_images(target, "/home/awhaley/CLionProjects/CameraNM500/images/People", CLASS_PEOPLE);
    train_images(target, "/home/awhaley/CLionProjects/CameraNM500/images/CarUphill", CLASS_CAR_UPHILL);
    train_images(target, "/home/awhaley/CLionProjects/CameraNM500/images/CarDownhill", CLASS_CAR_DOWNHILL);
    train_images(target, "/home/awhaley/CLionProjects/CameraNM500/images/TruckUphill", CLASS_CAR_UPHILL);

    tc_model_analysis(target);

    // Initialise the image matrices
    Mat roi = Mat(160, 160, CV_8U);
    Mat currentRoi = Mat(16, 16, CV_8UC1);
    Mat lastRoi = Mat(16, 16, CV_8UC1);
    Mat diff = Mat(16, 16, CV_8UC1);
    Mat preview = Mat(160, 160, CV_8UC1);

    // initialise the background (lastRoi) with current frame
    cap.read(frame);
    frame(Rect(420, 0, 160, 160)).copyTo(roi);
    resize(roi, lastRoi, cv::Size(16,16));
    cvtColor(lastRoi, lastRoi, cv::COLOR_RGB2GRAY);

    size_t t = 0;
    int tn = 0;
    time_t start, now, last_time;

    time(&start);
    time(&last_time);
    uint16_t last_cat = 0, cat, last_real_cat = 0;
    for (int i=1; i<=NUM_CLASSES; i++) totals[i] = 0;

    // Enter Frame read loop
    for (;;)
    {
        // wait for a new frame from camera and store it into 'frame'
        cap.read(frame);
        // check if we succeeded
        if (frame.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        // show live and wait for a key with timeout long enough to show images
        imshow("Live", frame);

        // Extract ROI
        frame(Rect(420, 0, 160, 160)).copyTo(roi);
        resize(roi, currentRoi, cv::Size(16,16));
        cvtColor(currentRoi, currentRoi, cv::COLOR_RGB2GRAY);

        // Calculate difference image - to show movement
        absdiff(currentRoi, lastRoi, diff);

        // Copy pixels from current to last
        memcpy(lastRoi.ptr(0), currentRoi.ptr(0), 256);
        resize(diff, preview, cv::Size(160,160));
        imshow("ROI", preview);

        t++;


        if (diff.isContinuous()) {
            nm_classify_req req;
            memcpy(req.vector, diff.ptr(0), 256);
            req.vector_size = 256;
            req.k = NUM_CLASSES;

            nm_classify(target, &req);
            if (req.status == NM_CLASSIFY_IDENTIFIED) {
                cat = req.category[0];
            }
            else cat = 0;

            time(&now);


            if ((cat > 0) && (cat != last_cat))
            {
                int elapsed = now - last_time;
                if ((cat > 1) && ((elapsed > 1) || (last_real_cat != cat)))
                {
                    totals[cat]++;
                    last_real_cat = cat;

                    printf("[%04d] (%4ds): ", tn, elapsed);
                    printf("%d - ", cat);
                    switch (cat) {
                        case CLASS_NOTHING:
                            printf("%-20s", "Nothing");
                            break;
                        case CLASS_PEOPLE:
                            printf("%-20s", "People");
                            imshow("People", roi);
                            break;
                        case CLASS_CAR_UPHILL:
                            printf("%-20s", "Car going uphill");
                            imshow("Uphill", roi);
                            break;
                        case CLASS_CAR_DOWNHILL:
                            printf("%-20s", "Car going downhill");
                            imshow("Downhill", roi);
                            break;
                        default:
                            break;
                    }
                    printf("[");
                    for (int i = 1; i <= NUM_CLASSES; i++) printf("%4d,", totals[i]);
                    printf("]\n");

                    // Save the file
                    char fn[256];
                    sprintf(fn, "roi_%d_%d.jpg", cat, totals[cat]);
                    imwrite(fn, roi);
                    sprintf(fn, "dif_%d_%d.jpg", cat, totals[cat]);
                    imwrite(fn, diff);

                    tn++;
                }
                last_time = now;
            }

            last_cat = cat;
        }



        // Quit if key pressed
        if (cv::waitKey(5) >= 0)
            break;
    }

    time_t end;
    time(&end);

    char fn[256];
    sprintf(fn, "img_%05d.jpg", 0);
    imwrite(fn, diff);
    tn++;


    printf("\n%2.1f fps.\n", (double) t / (double) (end - start) );

    // Free up NM500
    r = nm_power_save(target);
    printf("[set power_save mode] %d\n", r);

    r = nm_close(target);
    target = NULL; // just to make clean as a demo
    printf("[release resouces] %d\n", r);


    return 0;
}


void train_images(nm_device *device, char * folder, uint16_t category)
{
    DIR* FD;
    struct dirent* in_file;
    char fn[256];
    if (NULL == (FD = opendir (folder)))
    {
        fprintf(stderr, "Error : Failed to open input directory - %s\n", strerror(errno));
        return;
    }
    while ((in_file = readdir(FD)))
    {

        if (!strcmp (in_file->d_name, "."))
            continue;
        if (!strcmp (in_file->d_name, ".."))
            continue;
        sprintf(fn, "%s/%s", folder, in_file->d_name);
        printf("%s\n", fn);
        Mat img = imread(fn, 0);
        uint8_t v[256];
        memcpy(v, img.ptr(0), 256);
        tc_learn(device, v, 256, category);
    }

}


void tc_network_type(nm_device *device)
{
    uint16_t r;
    uint8_t network_type;
    r = nm_get_network_type(device, &network_type);
    printf("[tc_network_type]: Current Network Type is %s\n", (network_type == 0) ? "RBF" : "KNN");
    r = nm_set_network_type(device, KNN);
    printf("[tc_network_type]: Set Network Type as KNN\n");
    r = nm_get_network_type(device, &network_type);
    printf("[tc_network_type]: Current Network Type is %s\n", (network_type == 0) ? "RBF" : "KNN");
    r = nm_set_network_type(device, RBF);
    printf("[tc_network_type]: Set Network Type as RBF\n");
    r = nm_get_network_type(device, &network_type);
    printf("[tc_network_type]: Current Network Type is %s\n", (network_type == 0) ? "RBF" : "KNN");
}

void tc_read_neuron(nm_device *device, uint32_t nid, uint16_t vecter_size)
{
    uint16_t r;
    nm_neuron neuron;
    neuron.nid = nid;
    neuron.size = vecter_size;
    r = nm_read_neuron(device, &neuron);
    if (r == NM_SUCCESS) {
        printf("[tc_read_neuron] NID: %-5d, NCR: %-5d, CAT: %-5d, AIF: %-5d, MINIF: %-5d\n",
               neuron.nid, neuron.ncr, neuron.cat, neuron.aif, neuron.minif);
        for (int j = 0; j < neuron.size; j++) {
            printf("%d ", neuron.model[j]);
        }
        printf("\n");
    }
    else {
        printf("[tc_read_neuron] Error: Failed to read neuron. %d\n", r);
    }
}

void tc_read_neurons(nm_device *device, uint16_t vecter_size)
{
    uint16_t r;
    uint32_t neuron_count;
    r = nm_get_neuron_count(device, &neuron_count);
    if (r != NM_SUCCESS) {
        printf("[tc_read_neuron] Error: Failed to read the number of committed neurons. %d\n", r);
        return;
    }
    printf("[tc_read_neuron] neuronCount in ReadNeurons : %d\n", neuron_count);
    nm_neuron *neurons = (nm_neuron *)malloc(sizeof(nm_neuron) * neuron_count);
    neurons[0].size = vecter_size;

    r = nm_read_neurons(device, neurons, &neuron_count);
    if (r == NM_SUCCESS) {
        for (int i = 0; i < neuron_count; i++) {
            printf("[tc_read_neuron] NID: %-5d NCR: %-5d CAT: %-5d AIF: %-5d MINIF: %-5d\n", neurons[i].nid, neurons[i].ncr,
                   neurons[i].cat, neurons[i].aif, neurons[i].minif);
            for (int j = 0; j < vecter_size; j++) {
                printf("%d ", neurons[i].model[j]);
            }
            printf("\n");
        }
    }
    else {
        printf("[tc_read_neuron] Error: Failed to read neurons. %d\n", r);
    }
    free(neurons);
}

/*
    Read neurons and add one dummy neuron at the end of list.
    Total number of committed neuron will be increased.
*/
void tc_write_neurons(nm_device *device, uint16_t vecter_size)
{
    uint16_t r;
    uint32_t read_count, write_count;
    r = nm_get_neuron_count(device, &read_count);
    printf("[tc_write_neurons] neuron_count: %u\n", read_count);
    write_count = read_count + 1;
    nm_neuron *neurons = (nm_neuron *)malloc(sizeof(nm_neuron) * write_count);
    neurons[0].size = vecter_size;

    r = nm_read_neurons(device, neurons, &read_count);
    if (r == NM_SUCCESS) {
        neurons[write_count - 1].nid = write_count;
        neurons[write_count - 1].ncr = neurons[write_count - 2].ncr;
        neurons[write_count - 1].cat = 3;
        neurons[write_count - 1].aif = 10;
        neurons[write_count - 1].minif = neurons[write_count - 2].minif;
        neurons[write_count - 1].size = vecter_size;

        memset(neurons[write_count - 1].model, 0x00, 256);
        neurons[write_count - 1].model[0] = 3;
        neurons[write_count - 1].model[vecter_size - 1] = 2;

        r = nm_write_neurons(device, neurons, &write_count);
        if (r == NM_SUCCESS) {
            printf("[tc_write_neurons] neuron_count wrote: %u\n", write_count);
        }
        else {
            printf("[tc_write_neurons] Error: Failed to write neurons. %d\n", r);
        }
    }
    else {
        printf("[tc_write_neurons] Error: Failed to read neurons. %d\n", r);
    }

    free(neurons);
}

void tc_set_context(nm_device *device, uint8_t context, uint8_t norm, uint16_t minif, uint16_t maxif)
{
    uint16_t r;
    nm_context ctx;
    ctx.context = context;
    ctx.norm = norm;
    ctx.minif = minif;
    ctx.maxif = maxif;

    r = nm_set_context(device, &ctx);
    if (r == NM_SUCCESS) {
        printf("[tc_set_context] CONTEXT: %d NORM: %d MINIF: %d MAXIF: %d\n", context, norm, minif, maxif);
    }
    else {
        printf("[tc_set_context] Error: Failed to set context. %d\n", r);
    }
}

void tc_get_context(nm_device *device)
{
    uint16_t r;
    nm_context ctx;
    r = nm_get_context(device, &ctx);
    if (r == NM_SUCCESS) {
        printf("[tc_get_context] CONTEXT: %d, NORM: %d, MINIF: %d, MAXIF: %d\n", ctx.context, ctx.norm, ctx.minif, ctx.maxif);
    }
    else {
        printf("[tc_get_context] Error: Failed to get context. %d\n", r);
    }
}

void tc_classify(nm_device *device, uint8_t data, uint16_t size, uint8_t k)
{
    uint16_t r;
    nm_classify_req req;
    printf("\n[tc_classify] VECTOR: ");
    for (int i = 0; i < size; i++) {
        req.vector[i] = data;
        printf(" %-3d", data);
    }
    req.vector_size = size;
    req.k = k;
    printf(", k: %d\n", k);
    r = nm_classify(device, &req);
    if (r == NM_SUCCESS) {
        for (int i = 0; i < req.matched_count; i++) {
            printf("M[%d] NID: %-5d DISTANCE: %-5d CAT: %-5d\n", i, req.nid[i], req.distance[i], req.category[i]);
        }

        printf("NETWORK STATUS: %d\n", req.status);
    }
    else {
        printf("[tc_classify] Error: Failed to classify. %d\n", r);
    }
}

void tc_learn(nm_device *device, uint8_t* data, uint16_t size, uint16_t category)
{
    uint16_t r;
    nm_learn_req req;
    memset(&req, 0, sizeof req);
    // Set query_affected to 1 for retrieving the affected neurons
    req.query_affected = 1;
    printf("[tc_learn] VECTOR: ");
    for (int i = 0; i < size; i++) {
        req.vector[i] = data[i];
        //printf(" %-3d", data);
    }
    // Set the size of vector
    req.vector_size = size;
    req.category = category;
    printf(", CAT: %-5d ", category);
    r = nm_learn(device, &req);
    /*if (r == NM_SUCCESS) {
        printf(", RESULT: %d\n", req.status);

        if (req.query_affected == 1) {
            for (int i = 0; i < req.affected_count; i++) {
                printf("affected neuron nid: %d, aif: %d\n", req.affected_neurons[i].nid, req.affected_neurons[i].aif);
            }
        }
    }
    else {
        printf("[tc_learn] Error: Failed to learn. %d\n", r);
    } */
}

void tc_model_analysis(nm_device *device)
{
    nm_model_info mi;
    nm_get_model_info(device, &mi);
    printf("[tc_model_analysis] used: %d, max ctx: %d, max cat: %d\n", mi.count, mi.max_context, mi.max_category);

    nm_model_stat ms;
    ms.context = 1;
    // The size of array must be one greater than max category id,
    // because the category id starts with 1

    ms.histo_cat = (uint16_t *)malloc(sizeof(uint16_t) * (mi.max_category + 1));
    ms.histo_deg = (uint16_t *)malloc(sizeof(uint16_t) * (mi.max_category + 1));

    memset(ms.histo_cat, 0x00, sizeof(uint16_t) * (mi.max_category + 1));
    memset(ms.histo_deg, 0x00, sizeof(uint16_t) * (mi.max_category + 1));

    nm_get_model_stat(device, &ms);

    printf("[tc_model_analysis] ctx: %d, used: %d\n", ms.context, ms.count);
    printf("+----------+----------+----------+\n");
    printf("|%10s|%10s|%10s|\n", "cat", "count", "degen");
    printf("+----------+----------+----------+\n");
    for (int i = 1; i <= mi.max_category; i++) {
        printf("|%10d|%10d|%10d|\n", i, ms.histo_cat[i], ms.histo_deg[i]);
    }
    printf("+----------+----------+----------+\n");
    printf("\n");

    free(ms.histo_cat);
    free(ms.histo_deg);
}

void tc_learn_batch(nm_device *device, uint16_t size, uint32_t count)
{
    uint16_t r = NM_SUCCESS;

    printf("[tc_learn_batch] vector size: %d, vectors count: %d\n", size, count);

    nm_learn_batch_req batch;
    batch.iter_count = 10;
    batch.iter_result = (uint32_t *)malloc(sizeof(uint32_t) * batch.iter_count);
    // If iterable is set to 1,
    // it iterates over the given vectors until it is no longer learning.
    batch.iterable = 1;
    batch.vector_size = size;
    batch.vector_count = count;
    batch.vectors = (uint8_t *)malloc(sizeof(uint8_t) * (batch.vector_size * batch.vector_count));
    batch.categories = (uint16_t *)malloc(sizeof(uint16_t) * batch.vector_count);

    srand(time(NULL));
    uint8_t random_cat, random_vec;

    // Generate random vector
    for (int i = 0; i < batch.vector_count; i++) {
        // keep the same category for "already known" case
        if (i % 3 == 0) {
            do {
                // Get the random number
                random_cat = rand() % 50;
            } while (random_cat == 0);
        }
        printf("[tc_learn_batch] cat: %d, vector: ", random_cat);

        do {
            random_vec = rand() % 50;
        } while (random_vec == 0);

        for (int j = 0; j < batch.vector_size; j++) {
            batch.vectors[i * batch.vector_size + j] = random_vec;
            printf("%-3d ", random_vec);
        }
        batch.categories[i] = random_cat;
        printf("\n");
    }
    printf("\n");

    r = nm_learn_batch(device, &batch);
    printf("[tc_learn_batch] learning done ... %d\n", r);
    for (int i = 0; i < batch.iter_count; i++) {
        printf("[tc_learn_batch] epoch: %d, learned: %d\n", i, batch.iter_result[i]);
    }
    printf("\n");

    // Read neurons for debugging
    tc_read_neurons(device, batch.vector_size);

    free(batch.iter_result);
    free(batch.vectors);
    free(batch.categories);
}

void tc_clusterize(nm_device *device)
{
    uint16_t r = NM_SUCCESS;

    uint16_t vector_size = 3;
    uint32_t vector_count = 20;
    printf("[tc_clusterize] vector size: %d, vectors count: %d\n", vector_size, vector_count);

    nm_clusterize_req clu;
    // The value of initial category must be greater than 0
    clu.initial_category = 1;
    // If incrementof is set to 0, all of vector will be trained with same category
    clu.incrementof = 1;
    clu.vector_size = vector_size;
    clu.vector_count = vector_count;
    clu.vectors = (uint8_t *)malloc(sizeof(uint8_t) * (vector_size * vector_count));

    uint8_t random_vec;

    srand(time(NULL));

    // Generate random vectors
    for (int i = 0; i < vector_count; i++) {
        printf("[tc_clusterize] vector: ");

        do {
            // Get the random number
            random_vec = rand() % 50;
        } while (random_vec == 0);

        for (int j = 0; j < vector_size; j++) {
            clu.vectors[i * vector_size + j] = random_vec;
            printf("%-3d ", random_vec);
        }
        printf("\n");
    }
    printf("\n");

    // Need to set appropriate values for minimum/maximum influence field.
    tc_set_context(device, 1, L1, 2, 30);
    // Do clusterize
    r = nm_clusterize(device, &clu);
    // Read neurons for debugging
    tc_read_neurons(device, vector_size);

    free(clu.vectors);
}

void tc_save_model(nm_device *device, const char *path)
{
    uint16_t r = NM_SUCCESS;

    uint32_t neuron_count = 0;
    r = nm_get_neuron_count(device, &neuron_count);
    printf("[tc_save_model] neuron count: %d\n", neuron_count);

    nm_neuron *neurons = (nm_neuron *)malloc(sizeof(nm_neuron) * neuron_count);
    neurons[0].size = NEURON_MEMORY;

    r = nm_read_neurons(device, neurons, &neuron_count);

    r = nm_save_model(device, neurons, neuron_count, path);
    printf("[tc_save_model] save neurons to '%s' ... %d\n", path, r);

    free(neurons);
}

void tc_load_model(nm_device *device, const char *path)
{
    uint32_t neuron_count = 576;
    nm_neuron *neurons = (nm_neuron *)malloc(sizeof(nm_neuron) * neuron_count);
    neurons[0].size = NEURON_MEMORY;

    nm_load_model(device, neurons, &neuron_count, path);

    printf("[tc_load_model] neuron count: %d\n", neuron_count);
    for (int i = 0; i < neuron_count; i++) {
        printf("[tc_load_model] nid: %-5d ncr: %-5d cat: %-5d aif: %-5d minif: %-5d\n", neurons[i].nid, neurons[i].ncr, neurons[i].cat, neurons[i].aif, neurons[i].minif);
        for (int j = 0; j < NEURON_MEMORY; j++) {
            printf("%-3d ", neurons[i].model[j]);
        }
        printf("\n");
    }

    free(neurons);
}
