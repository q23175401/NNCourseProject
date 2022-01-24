from .base_layer import Layer
import numpy as np


class Conv2D(Layer):
    def __init__(self, input_shape: tuple, n_filters, kernel_size=[3, 3], strides=[2, 2], dilation_rate=[1, 1], padding='same'):
        self.input_shape = tuple(input_shape)
        # [x, y, channels]
        assert len(input_shape) == 3 and n_filters > 0

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.dialation_size = dilation_rate
        self.stride_size = strides
        self.padding = padding

        self.create_kernels()

    def create_kernels(self):
        self.kernels = np.random.uniform(
            0, 1, size=self.kernel_size+[self.input_shape[-1], self.n_filters])
        self.bias = np.zeros([self.n_filters])

        self.w_grads_mem = None
        self.b_grads_mem = None

    def get_weights(self):
        return [self.kernels, self.bias, self.w_grads_mem, self.b_grads_mem]

    def set_weights(self, weights):
        kernels, bias, w_grad_m, b_grad_m = weights
        assert np.shape(kernels) == np.shape(self.kernels) \
            and np.shape(bias) == np.shape(self.bias)

        self.kernels = np.array(kernels)
        self.bias = np.array(bias)

        self.w_grads_mem = [np.copy(m) for m in w_grad_m] if w_grad_m else None
        self.b_grads_mem = [np.copy(b) for b in b_grad_m] if b_grad_m else None

    def Forward(self, inputs):
        self.last_inputs = inputs
        shape = np.shape(inputs)
        assert len(shape) == 4 and shape[1:] == tuple(self.input_shape)

        # transform images one by one images = [batchsize, x, y, channel]
        result = []
        for im in inputs:
            out_im = self.transform_one_image(im)
            result.append(out_im)
        return np.array(result)

    def calculate_all_gradients(self, output_gradients):
        _, _, xex_offset, yex_offset = self.getValidBoundary()
        batch_size, out_width, out_height, out_depth = output_gradients.shape
        batch_size, in_width, in_height, _ = self.last_inputs.shape

        # calculate gradients of all weights, bias and correspond x
        b_grads = np.zeros_like(self.bias)
        w_grads = np.zeros_like(self.kernels)
        x_grads = np.zeros_like(self.last_inputs)

        # to run faster, run batches concurently
        # def calculate_gradients():
        def sum_up_gradients(batch_i):
            for out_d in range(out_depth):
                for out_x in range(out_width):
                    for out_y in range(out_height):
                        dE_dy = output_gradients[batch_i,
                                                 out_x, out_y, out_d]

                        # initial (inx, iny) point correspond to this (outx, outy)
                        conv_x = out_x * \
                            self.stride_size[0] - xex_offset
                        conv_y = out_y * \
                            self.stride_size[1] - yex_offset

                        # calculate x and y value filtered on inputs
                        # for all y output
                        #   calcualte dE/dw = Σ dE/dy * dy / dw, for all w in kernels
                        for fw in range(self.kernel_size[0]):
                            for fh in range(self.kernel_size[1]):
                                # correspond (inx, iny) to this (fw, fh) and (outx, outy)
                                in_x = conv_x + fw * \
                                    self.dialation_size[0]
                                in_y = conv_y + fh * \
                                    self.dialation_size[1]

                                if 0 <= in_x < in_width and 0 <= in_y < in_height:
                                    w_grads[fw, fh, :, out_d] += dE_dy * \
                                        self.last_inputs[batch_i,
                                                         in_x, in_y,
                                                         :]
                                    x_grads[batch_i, in_x, in_y, :] += dE_dy * \
                                        self.kernels[fw, fh, :, out_d]

                        b_grads[out_d] += dE_dy

        for bi in range(batch_size):
            sum_up_gradients(bi)

        return w_grads, b_grads, x_grads

    def Backward(self, output_gradients, optimizer):

        w_grads, b_grads, x_grads = self.calculate_all_gradients(
            output_gradients)

        w_result_grads, w_memory = optimizer.CalculateGradients(
            self.w_grads_mem, w_grads)
        b_result_grads, b_memory = optimizer.CalculateGradients(
            self.b_grads_mem, b_grads)

        self.w_grads_mem = w_memory
        self.b_grads_mem = b_memory

        self.kernels -= w_result_grads
        self.bias -= b_result_grads

        return x_grads

    def transform_one_image(self, input_image):
        valid_x_max, valid_y_max, xex_offset, yex_offset = self.getValidBoundary()

        # create empty output images [x, y, n_filtrers]
        output_width = \
            (valid_x_max + self.stride_size[0] - 1) // self.stride_size[0]
        output_height = \
            (valid_y_max + self.stride_size[1] - 1) // self.stride_size[1]
        out_image = np.zeros([output_width, output_height, self.n_filters])

        def calculate_filtered_image(filter_i):
            for out_xi in range(output_width):
                for out_yi in range(output_height):
                    image_xi = out_xi * self.stride_size[0] - xex_offset
                    image_yi = out_yi * self.stride_size[1] - yex_offset

                    out_image[out_xi, out_yi, filter_i] = self.getFilterValueAt(
                        image_xi, image_yi, filter_i, input_image)

        # calculate output images for every filters
        for fi in range(self.n_filters):
            calculate_filtered_image(fi)

        return out_image

    def transform_one_image_method2(self, input_image):
        output_image = None
        for fi in range(self.n_filters):
            filtered_image = None
            for depth in range(self.input_shape[2]):
                cor_im = self.correlation2D(
                    input_image[:, :, depth], self.kernels[:, :, depth, fi])

                filtered_image = filtered_image + cor_im if filtered_image is not None else cor_im
            filtered_image += self.bias[fi]

            final = filtered_image.reshape([*filtered_image.shape, 1])
            output_image = np.concatenate([output_image, final], axis=2) \
                if output_image is not None else final
        return output_image

    def getValidBoundary(self):
        '''
        get valid boundary that conv method applied on the image

        return:
            valid_x_max: filter value can be calculate at valid_x_max-1 - xex_offset
            valid_y_max: filter value can be calculate at valid_y_max-1 - yex_offset
            xex_offset: calculate filter value start from -xex_offset
            yex_offset: calculate filter value start from -yex_offset
        '''
        x_pos_max, y_pos_max, _ = self.input_shape
        filter_width, filter_height = self.kernel_size
        x_exceed = filter_width - 1   # x exceed if calculate value at x_pos_max-1
        y_exceed = filter_height - 1  # y exceed if calculate value at y_pos_max-1
        # calculate valid x and y position of every conv methods
        if self.padding == 'same':    # allow zero padding
            valid_x_max = x_pos_max
            valid_y_max = y_pos_max

            # calculate offset to minimize zeros used in "same" conv method
            xex_offset = \
                (x_exceed - (x_pos_max-1) % (self.stride_size[0])) // 2
            yex_offset = \
                (y_exceed - (y_pos_max-1) % (self.stride_size[1])) // 2

        elif self.padding == 'valid':  # no exceed allowed
            valid_x_max = x_pos_max - x_exceed
            valid_y_max = y_pos_max - y_exceed
            xex_offset = 0
            yex_offset = 0
        elif self.padding == 'full':
            valid_x_max = x_pos_max + x_exceed
            valid_y_max = y_pos_max + y_exceed
            xex_offset = x_exceed
            yex_offset = y_exceed
        return valid_x_max, valid_y_max, xex_offset, yex_offset

    def getFilterValueAt(self, img_xi, img_yi, fi, input_image):
        x_pos_max, y_pos_max, image_depth = self.input_shape

        total_sum = 0
        for fx in range(self.kernel_size[0]):
            for fy in range(self.kernel_size[1]):
                for di in range(image_depth):

                    x_pos = img_xi + fx * self.dialation_size[0]
                    y_pos = img_yi + fy * self.dialation_size[1]

                    image_value = input_image[x_pos, y_pos, di] \
                        if 0 <= x_pos < x_pos_max and 0 <= y_pos < y_pos_max else 0
                    filter_value = self.kernels[fx, fy, di, fi]
                    total_sum += filter_value * image_value

        return total_sum + self.bias[fi]

    def correlation2D(self, input2D, kernel2D, padding_value=0):
        x_pos_max, y_pos_max, _ = self.input_shape
        valid_x_max, valid_y_max, xex_offset, yex_offset = self.getValidBoundary()

        output = []
        for ix in range(-xex_offset, valid_x_max - xex_offset, self.stride_size[0]):
            a_row = []
            for iy in range(-yex_offset, valid_y_max-yex_offset, self.stride_size[1]):

                sum_value = 0
                for fx in range(self.kernel_size[0]):
                    for fy in range(self.kernel_size[1]):
                        imx = ix+fx*self.dialation_size[0]
                        imy = iy+fy*self.dialation_size[1]
                        im_value = input2D[imx, imy] \
                            if 0 <= imx < x_pos_max \
                            and 0 <= imy < y_pos_max else padding_value

                        sum_value += im_value*kernel2D[fx, fy]
                a_row.append(sum_value)

            output.append(a_row)
        return np.array(output)


if __name__ == "__main__":
    # seed = np.random.seed(1)
    image_shape = [64, 64, 3]
    image = np.random.uniform(0, 1, size=image_shape)

    # hyperparameters
    n_filters = 10
    kernel_len = 3
    stride_len = 2
    kernel_size = [kernel_len, kernel_len]
    stride_size = [stride_len, stride_len]
    dilation_size = [1, 1]
    padding = 'valid'

    myConv = Conv2D(input_shape=image_shape,
                    n_filters=n_filters,
                    kernel_size=kernel_size,
                    strides=stride_size,
                    dilation_rate=dilation_size,
                    padding=padding)

    # myout_image = myConv(np.array([image]))[0]
    # print(myout_image[:, :, 0])

    output_image = myConv.transform_one_image(image)
    output_image2 = myConv.transform_one_image_method2(image)

    print(np.allclose(output_image, output_image2))
