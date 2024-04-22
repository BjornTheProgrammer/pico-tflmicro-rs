#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
#include "wrapped.h"

extern "C" {
  const Model* getModel(const void* buffer, size_t len) {
    auto verifier = flatbuffers::Verifier((uint8_t *)buffer, len);
    if (!::tflite::VerifyModelBuffer(verifier)) {
        return NULL;
    }

    const tflite::Model* model = ::tflite::GetModel(buffer);
    return reinterpret_cast<const Model*>(model);
  }

  void destroyModel(const Model* model) {
    delete reinterpret_cast<const tflite::Model*>(model);
  }

  MicroInterpreter* getInterpreter(const Model* model, MicroMutableOpResolver* resolver, int kTensorArenaSize) {
    const tflite::Model* tflite_model = reinterpret_cast<const tflite::Model*>(model);
    tflite::MicroMutableOpResolver<128>* tflite_resolver = reinterpret_cast<tflite::MicroMutableOpResolver<128>*>(resolver);

    uint8_t tensor_arena[kTensorArenaSize];
    tflite::MicroInterpreter* interpreter = nullptr;

    static tflite::MicroInterpreter static_interpreter(tflite_model, *tflite_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    return reinterpret_cast<MicroInterpreter*>(interpreter);
}

  void destroyInterpreter(MicroInterpreter* interpreter) {
    delete reinterpret_cast<tflite::MicroInterpreter*>(interpreter);
  }

  TfLiteStatus allocateTensors(MicroInterpreter* interpreter) {
    tflite::MicroInterpreter* tflite_interpreter = reinterpret_cast<tflite::MicroInterpreter*>(interpreter);

    if (tflite_interpreter == nullptr) {
      printf("Got NULL!\n");
      return kTfLiteError;
    }

    printf("Allocated Tensors!\n");
    TfLiteStatus allocate_status = tflite_interpreter->AllocateTensors();
    return allocate_status;
  }

  MicroMutableOpResolver* createEmptyResolver() {
    static tflite::MicroMutableOpResolver<128>* resolver = new tflite::MicroMutableOpResolver<128>();
    return reinterpret_cast<MicroMutableOpResolver*>(resolver);
  }

  void destroyResolver(MicroMutableOpResolver* resolver) {
    delete reinterpret_cast<tflite::MicroMutableOpResolver<128>*>(resolver);
  }

  TfLiteStatus addCustomResolver(MicroMutableOpResolver* resolver, char* name, TFLMRegistration* registration) {
    static tflite::MicroMutableOpResolver<128>* tflite_resolver = reinterpret_cast<tflite::MicroMutableOpResolver<128>*>(resolver);
    return tflite_resolver->AddCustom(name, registration);
  }

  TfLiteStatus addResolver(MicroMutableOpResolver* resolver, AddableResolver resolverToAdd) {
    static tflite::MicroMutableOpResolver<128>* tflite_resolver = reinterpret_cast<tflite::MicroMutableOpResolver<128>*>(resolver);
    switch (resolverToAdd) {
      case AddAbs:
        return tflite_resolver->AddAbs();
        break;
      case AddAdd:
        return tflite_resolver->AddAdd();
        break;
      case AddAddN:
        return tflite_resolver->AddAddN();
        break;
      case AddArgMax:
        return tflite_resolver->AddArgMax();
        break;
      case AddArgMin:
        return tflite_resolver->AddArgMin();
        break;
      case AddAssignVariable:
        return tflite_resolver->AddAssignVariable();
        break;
      case AddAveragePool2D:
        return tflite_resolver->AddAveragePool2D();
        break;
      case AddBatchMatMul:
        return tflite_resolver->AddBatchMatMul();
        break;
      case AddBatchToSpaceNd:
        return tflite_resolver->AddBatchToSpaceNd();
        break;
      case AddBroadcastArgs:
        return tflite_resolver->AddBroadcastArgs();
        break;
      case AddBroadcastTo:
        return tflite_resolver->AddBroadcastTo();
        break;
      case AddCallOnce:
        return tflite_resolver->AddCallOnce();
        break;
      case AddCast:
        return tflite_resolver->AddCast();
        break;
      case AddCeil:
        return tflite_resolver->AddCeil();
        break;
      case AddCircularBuffer:
        return tflite_resolver->AddCircularBuffer();
        break;
      case AddConcatenation:
        return tflite_resolver->AddConcatenation();
        break;
      case AddConv2D:
        return tflite_resolver->AddConv2D();
        break;
      case AddCos:
        return tflite_resolver->AddCos();
        break;
      case AddCumSum:
        return tflite_resolver->AddCumSum();
        break;
      case AddDelay:
        return tflite_resolver->AddDelay();
        break;
      case AddDepthToSpace:
        return tflite_resolver->AddDepthToSpace();
        break;
      case AddDepthwiseConv2D:
        return tflite_resolver->AddDepthwiseConv2D();
        break;
      case AddDequantize:
        return tflite_resolver->AddDequantize();
        break;
      case AddDetectionPostprocess:
        return tflite_resolver->AddDetectionPostprocess();
        break;
      case AddDiv:
        return tflite_resolver->AddDiv();
        break;
      case AddEmbeddingLookup:
        return tflite_resolver->AddEmbeddingLookup();
        break;
      case AddEnergy:
        return tflite_resolver->AddEnergy();
        break;
      case AddElu:
        return tflite_resolver->AddElu();
        break;
      case AddEqual:
        return tflite_resolver->AddEqual();
        break;
      case AddEthosU:
        return tflite_resolver->AddEthosU();
        break;
      case AddExp:
        return tflite_resolver->AddExp();
        break;
      case AddExpandDims:
        return tflite_resolver->AddExpandDims();
        break;
      case AddFftAutoScale:
        return tflite_resolver->AddFftAutoScale();
        break;
      case AddFill:
        return tflite_resolver->AddFill();
        break;
      case AddFilterBank:
        return tflite_resolver->AddFilterBank();
        break;
      case AddFilterBankLog:
        return tflite_resolver->AddFilterBankLog();
        break;
      case AddFilterBankSquareRoot:
        return tflite_resolver->AddFilterBankSquareRoot();
        break;
      case AddFilterBankSpectralSubtraction:
        return tflite_resolver->AddFilterBankSpectralSubtraction();
        break;
      case AddFloor:
        return tflite_resolver->AddFloor();
        break;
      case AddFloorDiv:
        return tflite_resolver->AddFloorDiv();
        break;
      case AddFloorMod:
        return tflite_resolver->AddFloorMod();
        break;
      case AddFramer:
        return tflite_resolver->AddFramer();
        break;
      case AddFullyConnected:
        return tflite_resolver->AddFullyConnected();
        break;
      case AddGather:
        return tflite_resolver->AddGather();
        break;
      case AddGatherNd:
        return tflite_resolver->AddGatherNd();
        break;
      case AddGreater:
        return tflite_resolver->AddGreater();
        break;
      case AddGreaterEqual:
        return tflite_resolver->AddGreaterEqual();
        break;
      case AddHardSwish:
        return tflite_resolver->AddHardSwish();
        break;
      case AddIf:
        return tflite_resolver->AddIf();
        break;
      case AddIrfft:
        return tflite_resolver->AddIrfft();
        break;
      case AddL2Normalization:
        return tflite_resolver->AddL2Normalization();
        break;
      case AddL2Pool2D:
        return tflite_resolver->AddL2Pool2D();
        break;
      case AddLeakyRelu:
        return tflite_resolver->AddLeakyRelu();
        break;
      case AddLess:
        return tflite_resolver->AddLess();
        break;
      case AddLessEqual:
        return tflite_resolver->AddLessEqual();
        break;
      case AddLog:
        return tflite_resolver->AddLog();
        break;
      case AddLogicalAnd:
        return tflite_resolver->AddLogicalAnd();
        break;
      case AddLogicalNot:
        return tflite_resolver->AddLogicalNot();
        break;
      case AddLogicalOr:
        return tflite_resolver->AddLogicalOr();
        break;
      case AddLogistic:
        return tflite_resolver->AddLogistic();
        break;
      case AddLogSoftmax:
        return tflite_resolver->AddLogSoftmax();
        break;
      case AddMaximum:
        return tflite_resolver->AddMaximum();
        break;
      case AddMaxPool2D:
        return tflite_resolver->AddMaxPool2D();
        break;
      case AddMirrorPad:
        return tflite_resolver->AddMirrorPad();
        break;
      case AddMean:
        return tflite_resolver->AddMean();
        break;
      case AddMinimum:
        return tflite_resolver->AddMinimum();
        break;
      case AddMul:
        return tflite_resolver->AddMul();
        break;
      case AddNeg:
        return tflite_resolver->AddNeg();
        break;
      case AddNotEqual:
        return tflite_resolver->AddNotEqual();
        break;
      case AddOverlapAdd:
        return tflite_resolver->AddOverlapAdd();
        break;
      case AddPack:
        return tflite_resolver->AddPack();
        break;
      case AddPad:
        return tflite_resolver->AddPad();
        break;
      case AddPadV2:
        return tflite_resolver->AddPadV2();
        break;
      case AddPCAN:
        return tflite_resolver->AddPCAN();
        break;
      case AddPrelu:
        return tflite_resolver->AddPrelu();
        break;
      case AddQuantize:
        return tflite_resolver->AddQuantize();
        break;
      case AddReadVariable:
        return tflite_resolver->AddReadVariable();
        break;
      case AddReduceMax:
        return tflite_resolver->AddReduceMax();
        break;
      case AddRelu:
        return tflite_resolver->AddRelu();
        break;
      case AddRelu6:
        return tflite_resolver->AddRelu6();
        break;
      case AddReshape:
        return tflite_resolver->AddReshape();
        break;
      case AddResizeBilinear:
        return tflite_resolver->AddResizeBilinear();
        break;
      case AddResizeNearestNeighbor:
        return tflite_resolver->AddResizeNearestNeighbor();
        break;
      case AddRfft:
        return tflite_resolver->AddRfft();
        break;
      case AddRound:
        return tflite_resolver->AddRound();
        break;
      case AddRsqrt:
        return tflite_resolver->AddRsqrt();
        break;
      case AddSelectV2:
        return tflite_resolver->AddSelectV2();
        break;
      case AddShape:
        return tflite_resolver->AddShape();
        break;
      case AddSin:
        return tflite_resolver->AddSin();
        break;
      case AddSlice:
        return tflite_resolver->AddSlice();
        break;
      case AddSoftmax:
        return tflite_resolver->AddSoftmax();
        break;
      case AddSpaceToBatchNd:
        return tflite_resolver->AddSpaceToBatchNd();
        break;
      case AddSpaceToDepth:
        return tflite_resolver->AddSpaceToDepth();
        break;
      case AddSplit:
        return tflite_resolver->AddSplit();
        break;
      case AddSplitV:
        return tflite_resolver->AddSplitV();
        break;
      case AddSqueeze:
        return tflite_resolver->AddSqueeze();
        break;
      case AddSqrt:
        return tflite_resolver->AddSqrt();
        break;
      case AddSquare:
        return tflite_resolver->AddSquare();
        break;
      case AddSquaredDifference:
        return tflite_resolver->AddSquaredDifference();
        break;
      case AddStridedSlice:
        return tflite_resolver->AddStridedSlice();
        break;
      case AddStacker:
        return tflite_resolver->AddStacker();
        break;
      case AddSub:
        return tflite_resolver->AddSub();
        break;
      case AddSum:
        return tflite_resolver->AddSum();
        break;
      case AddSvdf:
        return tflite_resolver->AddSvdf();
        break;
      case AddTanh:
        return tflite_resolver->AddTanh();
        break;
      case AddTransposeConv:
        return tflite_resolver->AddTransposeConv();
        break;
      case AddTranspose:
        return tflite_resolver->AddTranspose();
        break;
      case AddUnpack:
        return tflite_resolver->AddUnpack();
        break;
      case AddUnidirectionalSequenceLSTM:
        return tflite_resolver->AddUnidirectionalSequenceLSTM();
        break;
      case AddVarHandle:
        return tflite_resolver->AddVarHandle();
        break;
      case AddWhile:
        return tflite_resolver->AddWhile();
        break;
      case AddWindow:
        return tflite_resolver->AddWindow();
        break;
      case AddZerosLike:
        return tflite_resolver->AddZerosLike();
        break;
      default:
        return kTfLiteError;
    }
  }

  TfLiteTensor* getTensorInput(MicroInterpreter* interpreter, size_t n) {
    tflite::MicroInterpreter* tflite_interpreter = reinterpret_cast<tflite::MicroInterpreter*>(interpreter);
    if (tflite_interpreter == nullptr) {
      return NULL;
    }

    return tflite_interpreter->input(n);
  }

  TfLiteStatus invokeInterpreter(MicroInterpreter* interpreter) {
    tflite::MicroInterpreter* tflite_interpreter = reinterpret_cast<tflite::MicroInterpreter*>(interpreter);
    if (tflite_interpreter == nullptr) {
      return kTfLiteError;
    }

    return tflite_interpreter->Invoke();
  }

  TfLiteTensor* getTensorOutput(MicroInterpreter* interpreter, size_t n) {
    tflite::MicroInterpreter* tflite_interpreter = reinterpret_cast<tflite::MicroInterpreter*>(interpreter);
    if (tflite_interpreter == nullptr) {
      return NULL;
    }

    return tflite_interpreter->output(n);
  }

}
