#include <torch/extension.h>
#include <torch/types.h>
#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <random>
#include <iostream>



namespace {

static const JOCTET EOI_BUFFER[1] = {JPEG_EOI};
struct torch_jpeg_error_mgr {
  struct jpeg_error_mgr pub; /* "public" fields */
  char jpegLastErrorMsg[JMSG_LENGTH_MAX]; /* error messages */
  jmp_buf setjmp_buffer; /* for return to caller */
};

using torch_jpeg_error_ptr = struct torch_jpeg_error_mgr*;

void torch_jpeg_error_exit(j_common_ptr cinfo) {
  /* cinfo->err really points to a torch_jpeg_error_mgr struct, so coerce
   * pointer */
  torch_jpeg_error_ptr myerr = (torch_jpeg_error_ptr)cinfo->err;

  /* Always display the message. */
  /* We could postpone this until after returning, if we chose. */
  // (*cinfo->err->output_message)(cinfo);
  /* Create the message */
  (*(cinfo->err->format_message))(cinfo, myerr->jpegLastErrorMsg);

  /* Return control to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
}


struct torch_jpeg_mgr {
  struct jpeg_source_mgr pub;
  const JOCTET* data;
  size_t len;
};

static void torch_jpeg_init_source(j_decompress_ptr cinfo) {}

static boolean torch_jpeg_fill_input_buffer(j_decompress_ptr cinfo) {
  // No more data.  Probably an incomplete image;  Raise exception.
  torch_jpeg_error_ptr myerr = (torch_jpeg_error_ptr)cinfo->err;
  strcpy(myerr->jpegLastErrorMsg, "Image is incomplete or truncated");
  longjmp(myerr->setjmp_buffer, 1);
}

static void torch_jpeg_skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
  torch_jpeg_mgr* src = (torch_jpeg_mgr*)cinfo->src;
  if (src->pub.bytes_in_buffer < (size_t)num_bytes) {
    // Skipping over all of remaining data;  output EOI.
    src->pub.next_input_byte = EOI_BUFFER;
    src->pub.bytes_in_buffer = 1;
  } else {
    // Skipping over only some of the remaining data.
    src->pub.next_input_byte += num_bytes;
    src->pub.bytes_in_buffer -= num_bytes;
  }
}

static void torch_jpeg_term_source(j_decompress_ptr cinfo) {}

static void torch_jpeg_set_source_mgr(
    j_decompress_ptr cinfo,
    const unsigned char* data,
    size_t len) {
  torch_jpeg_mgr* src;
  if (cinfo->src == 0) { // if this is first time;  allocate memory
    cinfo->src = (struct jpeg_source_mgr*)(*cinfo->mem->alloc_small)(
        (j_common_ptr)cinfo, JPOOL_PERMANENT, sizeof(torch_jpeg_mgr));
  }
  src = (torch_jpeg_mgr*)cinfo->src;
  src->pub.init_source = torch_jpeg_init_source;
  src->pub.fill_input_buffer = torch_jpeg_fill_input_buffer;
  src->pub.skip_input_data = torch_jpeg_skip_input_data;
  src->pub.resync_to_restart = jpeg_resync_to_restart; // default
  src->pub.term_source = torch_jpeg_term_source;
  // fill the buffers
  src->data = (const JOCTET*)data;
  src->len = len;
  src->pub.bytes_in_buffer = len;
  src->pub.next_input_byte = src->data;
}

} // namespace

torch::Tensor decode_and_crop_jpeg(const torch::Tensor& data,
                                   unsigned int crop_y,
                                   unsigned int crop_x,
                                   unsigned int crop_height,
                                   unsigned int crop_width) {
  struct jpeg_decompress_struct cinfo;
  struct torch_jpeg_error_mgr jerr;

  auto datap = data.data_ptr<uint8_t>();
  // Setup decompression structure
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = torch_jpeg_error_exit;
  /* Establish the setjmp return context for my_error_exit to use. */
  setjmp(jerr.setjmp_buffer);
  jpeg_create_decompress(&cinfo);
  torch_jpeg_set_source_mgr(&cinfo, datap, data.numel());

  // read info from header.
  jpeg_read_header(&cinfo, TRUE);

  int channels = cinfo.num_components;

  jpeg_start_decompress(&cinfo);

  int stride = crop_width * channels;
  auto tensor =
      torch::empty({int64_t(crop_height), int64_t(crop_width), channels}, torch::kU8);
  auto ptr = tensor.data_ptr<uint8_t>();

  unsigned int update_width = crop_width;
  jpeg_crop_scanline(&cinfo, &crop_x, &update_width);
  jpeg_skip_scanlines(&cinfo, crop_y);

  const int offset = (cinfo.output_width - crop_width) * channels;
  uint8_t* temp = nullptr;
  if(offset > 0) temp = new uint8_t[cinfo.output_width * channels];

  while (cinfo.output_scanline < crop_y + crop_height) {
    /* jpeg_read_scanlines expects an array of pointers to scanlines.
     * Here the array is only one element long, but you could ask for
     * more than one scanline at a time if that's more convenient.
     */
    if(offset>0){
      jpeg_read_scanlines(&cinfo, &temp, 1);
      memcpy(ptr, temp + offset, stride);
    }
    else
      jpeg_read_scanlines(&cinfo, &ptr, 1);
    ptr += stride;
  }
  if(offset > 0){
    delete[] temp;
    temp = nullptr;
  }
  if (cinfo.output_scanline < cinfo.output_height) {
    // Skip the rest of scanlines, required by jpeg_destroy_decompress.
    jpeg_skip_scanlines(&cinfo,
                        cinfo.output_height - crop_y - crop_height);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  return tensor.permute({2, 0, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("decode_and_crop_jpeg", &decode_and_crop_jpeg, "decode_and_crop_jpeg");
}
