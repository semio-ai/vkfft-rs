use std::sync::Arc;

use derive_more::{Display, Error};
use std::pin::Pin;
use vulkano::{
  buffer::BufferAccess,
  command_buffer::pool::UnsafeCommandPool,
  device::{Device, Queue},
  instance::PhysicalDevice,
  sync::Fence,
  SynchronizedVulkanObject, VulkanHandle, VulkanObject,
};

use std::ptr::addr_of_mut;

#[derive(Display, Debug, Error)]
pub enum BuildError {
  NoPhysicalDevice,
  NoDevice,
  NoQueue,
  NoFence,
  NoCommandPool,
  NoBuffer,
}

pub struct ConfigBuilder<'a> {
  fft_dim: u32,
  size: [u32; 3usize],

  physical_device: Option<PhysicalDevice<'a>>,
  device: Option<Arc<Device>>,
  queue: Option<Arc<Queue>>,
  fence: Option<&'a Fence>,
  command_pool: Option<Arc<UnsafeCommandPool>>,
  buffer: Option<BufferDesc>,
  input_buffer: Option<BufferDesc>,
  output_buffer: Option<BufferDesc>,
  temp_buffer: Option<BufferDesc>,
  kernel: Option<BufferDesc>,
  normalize: bool,
  zero_padding: [bool; 3usize],
  zeropad_left: [u32; 3usize],
  zeropad_right: [u32; 3usize],
  kernel_convolution: bool,
  convolution: bool,
  r2c: bool,
  coordinate_features: u32,
  disable_reorder_four_step: bool,
  batch_count: Option<u32>,
  precision: Precision,
  use_lut: bool,
  symmetric_kernel: bool,
  input_formatted: Option<bool>,
  output_formatted: Option<bool>,
}

impl<'a> ConfigBuilder<'a> {
  pub fn new() -> Self {
    Self {
      fft_dim: 1,
      size: [1, 1, 1],
      physical_device: None,
      device: None,
      queue: None,
      fence: None,
      command_pool: None,
      normalize: false,
      zero_padding: [false, false, false],
      zeropad_left: [0, 0, 0],
      zeropad_right: [0, 0, 0],
      kernel_convolution: false,
      r2c: false,
      coordinate_features: 1,
      disable_reorder_four_step: false,
      buffer: None,
      temp_buffer: None,
      input_buffer: None,
      output_buffer: None,
      batch_count: None,
      precision: Precision::Single,
      convolution: false,
      use_lut: false,
      symmetric_kernel: false,
      input_formatted: None,
      output_formatted: None,
      kernel: None,
    }
  }

  pub fn dim<const N: usize>(mut self, dim: &[u32; N]) -> Self {
    let len = dim.len();
    assert!(len <= 3);

    self.fft_dim = len as u32;
    if len > 0 {
      self.size[0] = dim[0];
    }
    if len > 1 {
      self.size[1] = dim[1];
    }
    if len > 2 {
      self.size[2] = dim[2];
    }
    self
  }

  pub fn physical_device(mut self, physical_device: PhysicalDevice<'a>) -> Self {
    self.physical_device = Some(physical_device);
    self
  }

  pub fn device(mut self, device: Arc<Device>) -> Self {
    self.device = Some(device);
    self
  }

  pub fn queue(mut self, queue: Arc<Queue>) -> Self {
    self.queue = Some(queue);
    self
  }

  pub fn command_pool(mut self, command_pool: Arc<UnsafeCommandPool>) -> Self {
    self.command_pool = Some(command_pool);
    self
  }

  pub fn fence(mut self, fence: &'a Fence) -> Self {
    self.fence = Some(fence);
    self
  }

  pub fn buffer<B>(mut self, buffer: B) -> Self
  where
    B: Into<BufferDesc>,
  {
    self.buffer = Some(buffer.into());
    self
  }

  pub fn temp_buffer<B>(mut self, temp_buffer: B) -> Self
  where
    B: Into<BufferDesc>,
  {
    self.temp_buffer = Some(temp_buffer.into());
    self
  }

  pub fn input_buffer<B>(mut self, input_buffer: B) -> Self
  where
    B: Into<BufferDesc>,
  {
    self.input_buffer = Some(input_buffer.into());
    self
  }

  pub fn output_buffer<B>(mut self, output_buffer: B) -> Self
  where
    B: Into<BufferDesc>,
  {
    self.output_buffer = Some(output_buffer.into());
    self
  }

  pub fn kernel<B>(mut self, kernel: B) -> Self
  where
    B: Into<BufferDesc>,
  {
    self.kernel = Some(kernel.into());
    self
  }

  pub fn normalize(mut self) -> Self {
    self.normalize = true;
    self
  }

  pub fn kernel_convolution(mut self) -> Self {
    self.kernel_convolution = true;
    self
  }

  pub fn symmetric_kernel(mut self) -> Self {
    self.symmetric_kernel = true;
    self
  }

  pub fn convolution(mut self) -> Self {
    self.convolution = true;
    self
  }

  pub fn r2c(mut self) -> Self {
    self.r2c = true;
    self
  }

  pub fn use_lut(mut self) -> Self {
    self.use_lut = true;
    self
  }

  pub fn coordinate_features(mut self, coordinate_features: u32) -> Self {
    self.coordinate_features = coordinate_features;
    self
  }

  pub fn disable_reorder_four_step(mut self) -> Self {
    self.disable_reorder_four_step = true;
    self
  }

  pub fn zero_padding<const N: usize>(mut self, zero_padding: &[bool; N]) -> Self {
    let len = zero_padding.len();
    assert!(len <= 3);

    if len > 0 {
      self.zero_padding[0] = zero_padding[0];
    }
    if len > 1 {
      self.zero_padding[1] = zero_padding[1];
    }
    if len > 2 {
      self.zero_padding[2] = zero_padding[2];
    }
    self
  }

  pub fn zeropad_left<const N: usize>(mut self, zeropad_left: &[u32; N]) -> Self {
    let len = zeropad_left.len();
    assert!(len <= 3);

    if len > 0 {
      self.zeropad_left[0] = zeropad_left[0];
    }
    if len > 1 {
      self.zeropad_left[1] = zeropad_left[1];
    }
    if len > 2 {
      self.zeropad_left[2] = zeropad_left[2];
    }
    self
  }

  pub fn zeropad_right<const N: usize>(mut self, zeropad_right: &[u32; N]) -> Self {
    let len = zeropad_right.len();
    assert!(len <= 3);

    if len > 0 {
      self.zeropad_right[0] = zeropad_right[0];
    }
    if len > 1 {
      self.zeropad_right[1] = zeropad_right[1];
    }
    if len > 2 {
      self.zeropad_right[2] = zeropad_right[2];
    }
    self
  }

  pub fn batch_count(mut self, batch_count: u32) -> Self {
    self.batch_count = Some(batch_count);
    self
  }

  pub fn input_formatted(mut self, input_formatted: bool) -> Self {
    self.input_formatted = Some(input_formatted);
    self
  }

  pub fn output_formatted(mut self, output_formatted: bool) -> Self {
    self.output_formatted = Some(output_formatted);
    self
  }

  pub fn build(self) -> Result<Config<'a>, BuildError> {
    let physical_device = match self.physical_device {
      Some(v) => v,
      None => return Err(BuildError::NoPhysicalDevice),
    };

    let device = match self.device {
      Some(v) => v,
      None => return Err(BuildError::NoDevice),
    };

    let queue = match self.queue {
      Some(v) => v,
      None => return Err(BuildError::NoQueue),
    };

    let fence = match self.fence {
      Some(v) => v,
      None => return Err(BuildError::NoFence),
    };

    let command_pool = match self.command_pool {
      Some(v) => v,
      None => return Err(BuildError::NoCommandPool),
    };

    Ok(Config {
      fft_dim: self.fft_dim,
      size: self.size,
      physical_device,
      device,
      queue,
      fence,
      command_pool,
      normalize: self.normalize,
      zero_padding: self.zero_padding,
      zeropad_left: self.zeropad_left,
      zeropad_right: self.zeropad_right,
      kernel_convolution: self.kernel_convolution,
      r2c: self.r2c,
      coordinate_features: self.coordinate_features,
      disable_reorder_four_step: self.disable_reorder_four_step,
      buffer: self.buffer,
      batch_count: self.batch_count,
      precision: self.precision,
      convolution: self.convolution,
      use_lut: self.use_lut,
      symmetric_kernel: self.symmetric_kernel,
      input_formatted: self.input_formatted,
      output_formatted: self.output_formatted,
      kernel: self.kernel,
      temp_buffer: self.temp_buffer,
      input_buffer: self.input_buffer,
      output_buffer: self.output_buffer,
    })
  }
}

pub enum Precision {
  /// Perform calculations in single precision (32-bit)
  Single,
  /// Perform calculations in double precision (64-bit)
  Double,
  /// Perform calculations in half precision (16-bit)
  Half,
  /// Use half precision only as input/output buffer. Input/Output have to be allocated as half,
  /// buffer/tempBuffer have to be allocated as float (out of place mode only).
  HalfMemory,
}

pub enum BufferDesc {
  Buffer(Arc<dyn BufferAccess>),
  BufferSize(usize),
}

impl<T> From<Arc<T>> for BufferDesc
where
  T: 'static + BufferAccess,
{
  fn from(value: Arc<T>) -> Self {
    Self::Buffer(value as Arc<dyn BufferAccess>)
  }
}

impl From<usize> for BufferDesc {
  fn from(value: usize) -> Self {
    Self::BufferSize(value)
  }
}

impl BufferDesc {
  pub fn size(&self) -> usize {
    match self {
      Self::Buffer(b) => b.size(),
      Self::BufferSize(b) => *b,
    }
  }

  pub fn as_buffer(&self) -> Option<&Arc<dyn BufferAccess>> {
    match self {
      Self::Buffer(b) => Some(b),
      Self::BufferSize(_) => None,
    }
  }

  pub fn as_buffer_size(&self) -> Option<&usize> {
    match self {
      Self::Buffer(_) => None,
      Self::BufferSize(b) => Some(b),
    }
  }
}

pub struct Config<'a> {
  pub fft_dim: u32,
  pub size: [u32; 3usize],

  pub physical_device: PhysicalDevice<'a>,
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
  pub fence: &'a Fence,
  pub command_pool: Arc<UnsafeCommandPool>,

  pub buffer: Option<BufferDesc>,
  pub input_buffer: Option<BufferDesc>,
  pub output_buffer: Option<BufferDesc>,
  pub temp_buffer: Option<BufferDesc>,
  pub kernel: Option<BufferDesc>,

  /// Normalize inverse transform
  pub normalize: bool,

  /// Don't read some data/perform computations if some input sequences are zeropadded for each axis
  pub zero_padding: [bool; 3usize],

  /// Specify start boundary of zero block in the system for each axis
  pub zeropad_left: [u32; 3usize],

  /// Specify end boundary of zero block in the system for each axis
  pub zeropad_right: [u32; 3usize],

  /// Specify if this application is used to create kernel for convolution, so it has the same properties
  pub kernel_convolution: bool,

  /// Perform convolution in this application (0 - off, 1 - on). Disables reorderFourStep parameter
  pub convolution: bool,

  /// Perform R2C/C2R decomposition
  pub r2c: bool,

  /// C - coordinate, or dimension of features vector. In matrix convolution - size of vector
  pub coordinate_features: u32,

  /// Disables unshuffling of four step algorithm. Requires `temp_buffer` allocation.
  pub disable_reorder_four_step: bool,

  /// Used to perform multiple batches of initial data
  pub batch_count: Option<u32>,

  pub precision: Precision,

  /// Switches from calculating sincos to using precomputed LUT tables
  pub use_lut: bool,

  /// Specify if kernel in 2x2 or 3x3 matrix convolution is symmetric
  pub symmetric_kernel: bool,

  /// specify if input buffer is padded - false is padded, true is not padded.
  /// For example if it is not padded for R2C if out-of-place mode is selected
  /// (only if numberBatches==1 and numberKernels==1)
  pub input_formatted: Option<bool>,

  /// specify if output buffer is padded - false is padded, true is not padded.
  /// For example if it is not padded for R2C if out-of-place mode is selected
  /// (only if numberBatches==1 and numberKernels==1)
  pub output_formatted: Option<bool>,
}

#[derive(Display, Debug, Error)]
pub enum ConfigError {
  InvalidConfig,
}

pub(crate) struct KeepAlive {
  pub device: Arc<Device>,
  pub queue: Arc<Queue>,
  pub command_pool: Arc<UnsafeCommandPool>,

  pub buffer: Option<Arc<dyn BufferAccess>>,
  pub input_buffer: Option<Arc<dyn BufferAccess>>,
  pub output_buffer: Option<Arc<dyn BufferAccess>>,
  pub temp_buffer: Option<Arc<dyn BufferAccess>>,
  pub kernel: Option<Arc<dyn BufferAccess>>,
}

#[repr(C)]
pub(crate) struct ConfigGuard {
  pub(crate) keep_alive: KeepAlive,
  pub(crate) config: vkfft_sys::VkFFTConfiguration,
  pub(crate) physical_device: vk_sys::PhysicalDevice,
  pub(crate) device: vk_sys::Device,
  pub(crate) queue: vk_sys::Queue,
  pub(crate) command_pool: vk_sys::CommandPool,
  pub(crate) fence: vk_sys::Fence,
  pub(crate) buffer_size: u64,
  pub(crate) buffer: Option<vk_sys::Buffer>,
  pub(crate) input_buffer_size: u64,
  pub(crate) input_buffer: Option<vk_sys::Buffer>,
  pub(crate) output_buffer_size: u64,
  pub(crate) output_buffer: Option<vk_sys::Buffer>,
  pub(crate) temp_buffer_size: u64,
  pub(crate) temp_buffer: Option<vk_sys::Buffer>,
  pub(crate) kernel_size: u64,
  pub(crate) kernel: Option<vk_sys::Buffer>,
}

impl<'a> Config<'a> {
  pub fn builder() -> ConfigBuilder<'a> {
    ConfigBuilder::new()
  }

  pub fn buffer_size(&self) -> usize {
    self.buffer.as_ref().map(|b| b.size()).unwrap_or(0)
  }

  pub fn buffer(&self) -> Option<&BufferDesc> {
    self.buffer.as_ref()
  }

  pub fn temp_buffer(&self) -> Option<&BufferDesc> {
    self.temp_buffer.as_ref()
  }

  pub fn input_buffer(&self) -> Option<&BufferDesc> {
    self.input_buffer.as_ref()
  }

  pub fn output_buffer(&self) -> Option<&BufferDesc> {
    self.output_buffer.as_ref()
  }

  pub fn kernel_convolution(&self) -> bool {
    self.kernel_convolution
  }

  pub fn symmetric_kernel(&self) -> bool {
    self.kernel_convolution
  }

  pub fn convolution(&self) -> bool {
    self.convolution
  }

  pub fn r2c(&self) -> bool {
    self.r2c
  }

  pub fn normalize(&self) -> bool {
    self.normalize
  }

  pub fn coordinate_features(&self) -> u32 {
    self.coordinate_features
  }

  pub fn batch_count(&self) -> Option<u32> {
    self.batch_count
  }

  pub fn use_lut(&self) -> bool {
    self.use_lut
  }

  pub(crate) fn as_sys(&self) -> Result<Pin<Box<ConfigGuard>>, ConfigError> {
    use std::mem::{transmute, zeroed};

    unsafe {
      let keep_alive = KeepAlive {
        device: self.device.clone(),
        buffer: self.buffer.as_ref().map(|b| b.as_buffer().cloned()).flatten(),
        input_buffer: self.input_buffer.as_ref().map(|b| b.as_buffer().cloned()).flatten(),
        output_buffer: self.output_buffer.as_ref().map(|b| b.as_buffer().cloned()).flatten(),
        kernel: self.kernel.as_ref().map(|b| b.as_buffer().cloned()).flatten(),
        command_pool: self.command_pool.clone(),
        queue: self.queue.clone(),
        temp_buffer: self.temp_buffer.as_ref().map(|b| b.as_buffer().cloned()).flatten()
      };

      let mut res = Box::pin(ConfigGuard {
        keep_alive, 
        config: zeroed(),
        physical_device: self.physical_device.internal_object(),
        device: self.device.internal_object().value() as usize,
        queue: self.queue.internal_object_guard().value() as usize,
        command_pool: self.command_pool.internal_object().value(),
        fence: self.fence.internal_object().value(),
        buffer_size: self.buffer.as_ref().map(|b| b.size()).unwrap_or(0) as u64,
        temp_buffer_size: self.temp_buffer.as_ref().map(|b| b.size()).unwrap_or(0) as u64,
        input_buffer_size: self.input_buffer.as_ref().map(|b| b.size()).unwrap_or(0) as u64,
        output_buffer_size: self.output_buffer.as_ref().map(|b| b.size()).unwrap_or(0) as u64,
        kernel_size: self.kernel.as_ref().map(|b| b.size()).unwrap_or(0) as u64,
        buffer: self
          .buffer
          .as_ref()
          .map(|b| b.as_buffer())
          .flatten()
          .map(|b| b.inner().buffer.internal_object().value()),
        temp_buffer: self
          .temp_buffer
          .as_ref()
          .map(|b| b.as_buffer())
          .flatten()
          .map(|b| b.inner().buffer.internal_object().value()),
        input_buffer: self
          .input_buffer
          .as_ref()
          .map(|b| b.as_buffer())
          .flatten()
          .map(|b| b.inner().buffer.internal_object().value()),
        output_buffer: self
          .output_buffer
          .as_ref()
          .map(|b| b.as_buffer())
          .flatten()
          .map(|b| b.inner().buffer.internal_object().value()),
        kernel: self
          .kernel
          .as_ref()
          .map(|b| b.as_buffer())
          .flatten()
          .map(|b| b.inner().buffer.internal_object().value()),
      });

      res.config.FFTdim = self.fft_dim;
      res.config.size = self.size;

      res.config.physicalDevice = transmute(addr_of_mut!(res.physical_device));
      res.config.device = transmute(addr_of_mut!(res.device));
      res.config.queue = transmute(addr_of_mut!(res.queue));
      res.config.commandPool = transmute(addr_of_mut!(res.command_pool));
      res.config.fence = transmute(addr_of_mut!(res.fence));
      res.config.normalize = self.normalize.into();

      if res.kernel_size != 0 {
        res.config.kernelNum = 1;
        res.config.kernelSize = transmute(addr_of_mut!(res.kernel_size));
      }

      if let Some(t) = &res.kernel {
        println!("K: {:#0x}", t);
        res.config.kernel = transmute(t);
      }

      if res.buffer_size != 0 {
        res.config.bufferNum = 1;
        res.config.bufferSize = transmute(addr_of_mut!(res.buffer_size));
      }

      if let Some(t) = &res.buffer {
        println!("B: {:#0x}", *t);
        res.config.buffer = transmute(t);
      }

      if res.temp_buffer_size != 0 {
        res.config.tempBufferNum = 1;
        res.config.tempBufferSize = transmute(addr_of_mut!(res.temp_buffer_size));
      }

      if let Some(t) = &res.temp_buffer {
        println!("T: {:#0x}", *t);
        res.config.tempBuffer = transmute(t);
      }

      if res.input_buffer_size != 0 {
        res.config.inputBufferNum = 1;
        res.config.inputBufferSize = transmute(addr_of_mut!(res.input_buffer_size));
      }

      if let Some(t) = &res.input_buffer {
        println!("I: {:#0x}", *t);
        res.config.inputBuffer = transmute(t);
      }

      if res.output_buffer_size != 0 {
        res.config.outputBufferNum = 1;
        res.config.outputBufferSize = transmute(addr_of_mut!(res.output_buffer_size));

      }

      if let Some(t) = &res.output_buffer {
        println!("O: {:#0x}", *t);
        res.config.outputBuffer = transmute(t);
      }

      res.config.performZeropadding[0] = self.zero_padding[0].into();
      res.config.performZeropadding[1] = self.zero_padding[1].into();
      res.config.performZeropadding[2] = self.zero_padding[2].into();

      res.config.fft_zeropad_left = self.zeropad_left;
      res.config.fft_zeropad_right = self.zeropad_right;

      res.config.kernelConvolution = self.kernel_convolution.into();
      res.config.performR2C = self.r2c.into();
      res.config.coordinateFeatures = self.coordinate_features;
      res.config.disableReorderFourStep = self.disable_reorder_four_step.into();

      res.config.symmetricKernel = self.symmetric_kernel.into();

      if let Some(input_formatted) = self.input_formatted {
        res.config.isInputFormatted = input_formatted.into();
      }

      if let Some(output_formatted) = self.output_formatted {
        res.config.isOutputFormatted = output_formatted.into();
      }

      match self.precision {
        Precision::Double => {
          res.config.doublePrecision = true.into();
        }
        Precision::Half => res.config.halfPrecision = true.into(),
        Precision::HalfMemory => {
          res.config.halfPrecisionMemoryOnly = true.into();

          if let Some(false) = self.input_formatted {
            return Err(ConfigError::InvalidConfig);
          }

          if let Some(false) = self.output_formatted {
            return Err(ConfigError::InvalidConfig);
          }

          res.config.isInputFormatted = true.into();
          res.config.isOutputFormatted = true.into();
        }
        _ => {}
      }

      if let Some(batch_count) = &self.batch_count {
        res.config.numberBatches = *batch_count;
      }

      Ok(res)
    }
  }
}
