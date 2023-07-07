#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <fstream>
#include <map>
#include <chrono>

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

// stuff we know about the network and the input/output blobs
static const int INPUT_LEN= 10;
static const int OUTPUT_SIZE = 10;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

static Logger gLogger;

IElementWiseLayer* addL2Norm(INetworkDefinition* network,  ITensor& input, const float eps){
    /* get l2 normalization*/
    const static float pval1[3]{0.0, 1.0, 2.0};
    Weights wshift1{DataType::kFLOAT, pval1, 1};
    Weights wscale1{DataType::kFLOAT, pval1+1, 1};
    Weights wpower1{DataType::kFLOAT, pval1+2, 1};

    IScaleLayer* scale1 = network->addScale(input,
        ScaleMode::kUNIFORM,
        wshift1,  
        wscale1,  
        wpower1);
    assert(scale1);

    IReduceLayer* reduce1 = network->addReduce(*scale1->getOutput(0),
        ReduceOperation::kSUM, 
        7,   //  reduce C,H,W dims --> bitmask: 2^0 + 2^1 + 2^2=7
        true); 
    assert(reduce1);

    const static float pval2[3]{eps, 1.0, 0.5};
    Weights wshift2{DataType::kFLOAT, pval2, 1};
    Weights wscale2{DataType::kFLOAT, pval2+1, 1};
    Weights wpower2{DataType::kFLOAT, pval2+2, 1};   

    IScaleLayer* scale2 = network->addScale(*reduce1->getOutput(0),
        ScaleMode::kUNIFORM,
        wshift2,  
        wscale2,  
        wpower2);
    assert(scale2);  

    IElementWiseLayer* ew = network ->addElementWise(input, 
        *scale2->getOutput(0), 
        ElementWiseOperation::kDIV);
    assert(ew);  
    return ew;
}


// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{   
    // create network with implicit batch size
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {10, 1, 1} with name INPUT_BLOB_NAME, this is a feature with length=10
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{INPUT_LEN,1,1});
    assert(data);

    /* get l2 normalization*/
    float eps = 1e-5f;
    IElementWiseLayer* L2Norm = addL2Norm(network, *data, eps);

    L2Norm->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    std::cout << "set name out" << std::endl;
    network->markOutput(*L2Norm->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1 << 20);
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "build out" << std::endl;

    // Don't need the network any more
    network->destroy();

    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize *  INPUT_LEN* sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_LEN* sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./l2norm -s   // serialize model to plan file" << std::endl;
        std::cerr << "./l2norm -d   // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("l2norm.engine", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("l2norm.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }


    // set input feature = list(range(10))
    float data[ INPUT_LEN];
    for (int i = 0; i < INPUT_LEN; i++)
        data[i] = i;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float prob[OUTPUT_SIZE];
    for (int i = 0; i < 10; i++) {
        auto start = std::chrono::system_clock::now();
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        // std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nOutput:\n\n";
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++)
    {
        std::cout << prob[i] << ", ";
        if ((i+1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;


    std::cout << "\nInput:\n\n";
    for (unsigned int i = 0; i < INPUT_LEN; i++)
    {
        std::cout << data[i] << ", ";
        if ((i+1) % 10 == 0) std::cout << std::endl;
    }
    std::cout << std::endl;

    return 0;
}