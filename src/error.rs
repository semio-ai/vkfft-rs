use std::convert::{TryFrom, TryInto};

use derive_more::{Display, Error};

use crate::{app::LaunchError, config::ConfigError};

#[derive(Display, Debug, Error)]
pub enum Error {
  InvalidPhysicalDevice,
  InvalidDevice,
  InvalidQueue,
  InvalidCommandPool,
  InvalidFence,
  OnlyForwardFftInitialized,
  OnlyInverseFftInitialized,
  InvalidContext,
  InvalidPlatform,
  EmptyFftDim,
  EmptySize,
  EmptyBufferSize,
  EmptyBuffer,
  EmptyTempBufferSize,
  EmptyTempBuffer,
  EmptyInputBufferSize,
  EmptyInputBuffer,
  EmptyOutputBufferSize,
  EmptyOutputBuffer,
  EmptyKernelSize,
  EmptyKernel,
  UnsupportedRadix,
  UnsupportedFftLength,
  UnsupportedFftLengthR2C,
  FailedToAllocate,
  FailedToMapMemory,
  FailedToAllocateCommandBuffers,
  FailedToBeginCommandBuffer,
  FailedToEndCommandBuffer,
  FailedToSubmitQueue,
  FailedToWaitForFences,
  FailedToResetFences,
  FailedToCreateDescriptorPool,
  FailedToCreatedDescriptorSetLayout,
  FailedToAllocateDescriptorSets,
  FailedToCreatePipelineLayout,
  FailedShaderPreprocess,
  FailedShaderParse,
  FailedShaderLink,
  FailedSpirvGenerate,
  FailedToCreateShaderModule,
  FailedToCreateInstance,
  FailedToSetupDebugMessenger,
  FailedToFindPhysicalDevice,
  FailedToCreateDevice,
  FailedToCreateFence,
  FailedToCreateCommandPool,
  FailedToCreateBuffer,
  FailedToAllocateMemory,
  FailedToBindBufferMemory,
  FailedToFindMemory,
  FailedToSynchronize,
  FailedToCopy,
  FailedToCreateProgram,
  FailedToCompileProgram,
  FailedToGetCodeSize,
  FailedToGetCode,
  FailedToDestroyProgram,
  FailedToLoadModule,
  FailedToGetFunction,
  FailedToSetDynamicSharedMemory,
  FailedToModuleGetGlobal,
  FailedToLaunchKernel,
  FailedToEventRecord,
  FailedToAddNameExpression,
  FailedToInitialize,
  FailedToSetDeviceId,
  FailedToGetDevice,
  FailedToCreateContext,
  FailedToCreatePipeline,
  FailedToSetKernelArg,
  FailedToCreateCommandQueue,
  FailedToReleaseCommandQueue,
  FailedToEnumerateDevices,
  Config(ConfigError),
  Launch(LaunchError)
}

impl TryFrom<vkfft_sys::VkFFTResult> for Error {
  type Error = ();

  #[allow(non_upper_case_globals)]
  fn try_from(value: vkfft_sys::VkFFTResult) -> std::result::Result<Self, Self::Error> {
    use vkfft_sys::*;

    match value {
      VkFFTResult_VKFFT_ERROR_INVALID_PHYSICAL_DEVICE => Ok(Self::InvalidPhysicalDevice),
      VkFFTResult_VKFFT_ERROR_INVALID_DEVICE => Ok(Self::InvalidDevice),
      VkFFTResult_VKFFT_ERROR_INVALID_QUEUE => Ok(Self::InvalidQueue),
      VkFFTResult_VKFFT_ERROR_INVALID_COMMAND_POOL => Ok(Self::InvalidCommandPool),
      VkFFTResult_VKFFT_ERROR_INVALID_FENCE => Ok(Self::InvalidFence),
      VkFFTResult_VKFFT_ERROR_ONLY_FORWARD_FFT_INITIALIZED => Ok(Self::OnlyForwardFftInitialized),
      VkFFTResult_VKFFT_ERROR_ONLY_INVERSE_FFT_INITIALIZED => Ok(Self::OnlyInverseFftInitialized),
      VkFFTResult_VKFFT_ERROR_INVALID_CONTEXT => Ok(Self::InvalidContext),
      VkFFTResult_VKFFT_ERROR_INVALID_PLATFORM => Ok(Self::InvalidPlatform),
      VkFFTResult_VKFFT_ERROR_EMPTY_FFTdim => Ok(Self::EmptyFftDim),
      VkFFTResult_VKFFT_ERROR_EMPTY_size => Ok(Self::EmptySize),
      VkFFTResult_VKFFT_ERROR_EMPTY_bufferSize => Ok(Self::EmptyBufferSize),
      VkFFTResult_VKFFT_ERROR_EMPTY_buffer => Ok(Self::EmptyBuffer),
      VkFFTResult_VKFFT_ERROR_EMPTY_tempBufferSize => Ok(Self::EmptyTempBufferSize),
      VkFFTResult_VKFFT_ERROR_EMPTY_tempBuffer => Ok(Self::EmptyTempBuffer),
      VkFFTResult_VKFFT_ERROR_EMPTY_inputBufferSize => Ok(Self::EmptyInputBufferSize),
      VkFFTResult_VKFFT_ERROR_EMPTY_inputBuffer => Ok(Self::EmptyInputBuffer),
      VkFFTResult_VKFFT_ERROR_EMPTY_outputBufferSize => Ok(Self::EmptyOutputBufferSize),
      VkFFTResult_VKFFT_ERROR_EMPTY_outputBuffer => Ok(Self::EmptyOutputBuffer),
      VkFFTResult_VKFFT_ERROR_EMPTY_kernelSize => Ok(Self::EmptyKernelSize),
      VkFFTResult_VKFFT_ERROR_EMPTY_kernel => Ok(Self::EmptyKernel),
      VkFFTResult_VKFFT_ERROR_UNSUPPORTED_RADIX => Ok(Self::UnsupportedRadix),
      VkFFTResult_VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH => Ok(Self::UnsupportedFftLength),
      VkFFTResult_VKFFT_ERROR_UNSUPPORTED_FFT_LENGTH_R2C => Ok(Self::UnsupportedFftLengthR2C),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_ALLOCATE => Ok(Self::FailedToAllocate),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_MAP_MEMORY => Ok(Self::FailedToMapMemory),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_ALLOCATE_COMMAND_BUFFERS => {
        Ok(Self::FailedToAllocateCommandBuffers)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_BEGIN_COMMAND_BUFFER => {
        Ok(Self::FailedToBeginCommandBuffer)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_END_COMMAND_BUFFER => Ok(Self::FailedToEndCommandBuffer),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_SUBMIT_QUEUE => Ok(Self::FailedToSubmitQueue),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_WAIT_FOR_FENCES => Ok(Self::FailedToWaitForFences),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_RESET_FENCES => Ok(Self::FailedToResetFences),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_POOL => {
        Ok(Self::FailedToCreateDescriptorPool)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT => {
        Ok(Self::FailedToCreatedDescriptorSetLayout)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_ALLOCATE_DESCRIPTOR_SETS => {
        Ok(Self::FailedToAllocateDescriptorSets)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE_LAYOUT => {
        Ok(Self::FailedToCreatePipelineLayout)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_SHADER_PREPROCESS => Ok(Self::FailedShaderPreprocess),
      VkFFTResult_VKFFT_ERROR_FAILED_SHADER_PARSE => Ok(Self::FailedShaderParse),
      VkFFTResult_VKFFT_ERROR_FAILED_SHADER_LINK => Ok(Self::FailedShaderLink),
      VkFFTResult_VKFFT_ERROR_FAILED_SPIRV_GENERATE => Ok(Self::FailedSpirvGenerate),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_SHADER_MODULE => {
        Ok(Self::FailedToCreateShaderModule)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_INSTANCE => Ok(Self::FailedToCreateInstance),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_SETUP_DEBUG_MESSENGER => {
        Ok(Self::FailedToSetupDebugMessenger)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_FIND_PHYSICAL_DEVICE => {
        Ok(Self::FailedToFindPhysicalDevice)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_DEVICE => Ok(Self::FailedToCreateDevice),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_FENCE => Ok(Self::FailedToCreateFence),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_POOL => Ok(Self::FailedToCreateCommandPool),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_BUFFER => Ok(Self::FailedToCreateBuffer),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_ALLOCATE_MEMORY => Ok(Self::FailedToAllocateMemory),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_BIND_BUFFER_MEMORY => Ok(Self::FailedToBindBufferMemory),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_FIND_MEMORY => Ok(Self::FailedToFindMemory),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_SYNCHRONIZE => Ok(Self::FailedToSynchronize),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_COPY => Ok(Self::FailedToCopy),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_PROGRAM => Ok(Self::FailedToCreateProgram),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_COMPILE_PROGRAM => Ok(Self::FailedToCompileProgram),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_GET_CODE_SIZE => Ok(Self::FailedToGetCodeSize),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_GET_CODE => Ok(Self::FailedToGetCode),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_DESTROY_PROGRAM => Ok(Self::FailedToDestroyProgram),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_LOAD_MODULE => Ok(Self::FailedToLoadModule),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_GET_FUNCTION => Ok(Self::FailedToGetFunction),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_SET_DYNAMIC_SHARED_MEMORY => {
        Ok(Self::FailedToSetDynamicSharedMemory)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_MODULE_GET_GLOBAL => Ok(Self::FailedToModuleGetGlobal),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_LAUNCH_KERNEL => Ok(Self::FailedToLaunchKernel),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_EVENT_RECORD => Ok(Self::FailedToEventRecord),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_ADD_NAME_EXPRESSION => Ok(Self::FailedToAddNameExpression),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_INITIALIZE => Ok(Self::FailedToInitialize),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_SET_DEVICE_ID => Ok(Self::FailedToSetDeviceId),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_GET_DEVICE => Ok(Self::FailedToGetDevice),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_CONTEXT => Ok(Self::FailedToCreateContext),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_PIPELINE => Ok(Self::FailedToCreatePipeline),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_SET_KERNEL_ARG => Ok(Self::FailedToSetKernelArg),
      VkFFTResult_VKFFT_ERROR_FAILED_TO_CREATE_COMMAND_QUEUE => {
        Ok(Self::FailedToCreateCommandQueue)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_RELEASE_COMMAND_QUEUE => {
        Ok(Self::FailedToReleaseCommandQueue)
      }
      VkFFTResult_VKFFT_ERROR_FAILED_TO_ENUMERATE_DEVICES => Ok(Self::FailedToEnumerateDevices),
      _ => Err(()),
    }
  }
}

impl From<ConfigError> for Error {
  fn from(e: ConfigError) -> Self {
    Self::Config(e)
  }
}

impl From<LaunchError> for Error {
  fn from(e: LaunchError) -> Self {
    Self::Launch(e)
  }
}

pub(crate) fn check_error(result: vkfft_sys::VkFFTResult) -> Result<()> {
  match result.try_into() {
    Ok(err) => Err(err),
    Err(_) => Ok(()),
  }
}

pub type Result<T> = std::result::Result<T, Error>;
