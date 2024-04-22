#ifndef WRAPPED_TF_HEADERS
#define WRAPPED_TF_HEADERS
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MicroInterpreter MicroInterpreter;
typedef struct Model Model;
typedef struct MicroMutableOpResolver MicroMutableOpResolver;

enum AddableResolver {
  AddAbs,
  AddAdd,
  AddAddN,
  AddArgMax,
  AddArgMin,
  AddAssignVariable,
  AddAveragePool2D,
  AddBatchMatMul,
  AddBatchToSpaceNd,
  AddBroadcastArgs,
  AddBroadcastTo,
  AddCallOnce,
  AddCast,
  AddCeil,
  AddCircularBuffer,
  AddConcatenation,
  AddConv2D,
  AddCos,
  AddCumSum,
  AddDelay,
  AddDepthToSpace,
  AddDepthwiseConv2D,
  AddDequantize,
  AddDetectionPostprocess,
  AddDiv,
  AddEmbeddingLookup,
  AddEnergy,
  AddElu,
  AddEqual,
  AddEthosU,
  AddExp,
  AddExpandDims,
  AddFftAutoScale,
  AddFill,
  AddFilterBank,
  AddFilterBankLog,
  AddFilterBankSquareRoot,
  AddFilterBankSpectralSubtraction,
  AddFloor,
  AddFloorDiv,
  AddFloorMod,
  AddFramer,
  AddFullyConnected,
  AddGather,
  AddGatherNd,
  AddGreater,
  AddGreaterEqual,
  AddHardSwish,
  AddIf,
  AddIrfft,
  AddL2Normalization,
  AddL2Pool2D,
  AddLeakyRelu,
  AddLess,
  AddLessEqual,
  AddLog,
  AddLogicalAnd,
  AddLogicalNot,
  AddLogicalOr,
  AddLogistic,
  AddLogSoftmax,
  AddMaximum,
  AddMaxPool2D,
  AddMirrorPad,
  AddMean,
  AddMinimum,
  AddMul,
  AddNeg,
  AddNotEqual,
  AddOverlapAdd,
  AddPack,
  AddPad,
  AddPadV2,
  AddPCAN,
  AddPrelu,
  AddQuantize,
  AddReadVariable,
  AddReduceMax,
  AddRelu,
  AddRelu6,
  AddReshape,
  AddResizeBilinear,
  AddResizeNearestNeighbor,
  AddRfft,
  AddRound,
  AddRsqrt,
  AddSelectV2,
  AddShape,
  AddSin,
  AddSlice,
  AddSoftmax,
  AddSpaceToBatchNd,
  AddSpaceToDepth,
  AddSplit,
  AddSplitV,
  AddSqueeze,
  AddSqrt,
  AddSquare,
  AddSquaredDifference,
  AddStridedSlice,
  AddStacker,
  AddSub,
  AddSum,
  AddSvdf,
  AddTanh,
  AddTransposeConv,
  AddTranspose,
  AddUnpack,
  AddUnidirectionalSequenceLSTM,
  AddVarHandle,
  AddWhile,
  AddWindow,
  AddZerosLike
};

const Model* getModel(const void* buffer, size_t len);
void destroyModel(const Model* model);
MicroInterpreter* getInterpreter(const Model* model, MicroMutableOpResolver* resolver, int kTensorArenaSize);
void destroyInterpreter(MicroInterpreter* interpreter);
TfLiteStatus allocateTensors(MicroInterpreter* interpreter);
MicroMutableOpResolver* createEmptyResolver();
void destroyResolver(MicroMutableOpResolver* resolver);
TfLiteStatus addCustomResolver(MicroMutableOpResolver* resolver, char* name, TFLMRegistration* registration);
TfLiteStatus addResolver(MicroMutableOpResolver* resolver, AddableResolver resolverToAdd);
TfLiteTensor* getTensorInput(MicroInterpreter* interpreter, size_t n);
TfLiteStatus invokeInterpreter(MicroInterpreter* interpreter);
TfLiteTensor* getTensorOutput(MicroInterpreter* interpreter, size_t n);


#ifdef __cplusplus
}
#endif
#endif
