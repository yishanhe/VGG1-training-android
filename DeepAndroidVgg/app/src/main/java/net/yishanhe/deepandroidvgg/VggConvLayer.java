package net.yishanhe.deepandroidvgg;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.RectifedLinear;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

/**
 * Created by syi on 5/17/17.
 */

public class VggConvLayer {


    private String name;

    int miniBatch;
    int inDepth;
    int inH;
    int inW;
    int outDepth;
    int kH;
    int kW;
    int outH;
    int outW;

    // parameter
    public INDArray input; // a view from outside, 4d miniBatch, inDepth, inH, inW,  NCHW

    // internal NDArrays
    private INDArray paramsFlattened;  // W+b
    private INDArray gradientsFlattened; // dL/dW + b

    // NDArray views
    private INDArray weightGradientsView;
    private INDArray biasGradientsView;

    private INDArray weightsView;
    private INDArray biasesView;


    // internal
    private INDArray z;
    private INDArray im2col2d;
    private INDArray epsNext2d;
    // static
    static int[] kernels = new int[]{3, 3};
    static int[] strides = new int[]{1, 1};
    static int[] pad = new int[]{1, 1};


    public VggConvLayer(String name, int miniBatch, int inDepth, int inH, int inW, int outDepth, int kH, int kW) {
        this.name = name;
        this.miniBatch = miniBatch;
        this.inDepth = inDepth;
        this.inH = inH;
        this.inW = inW;
        this.outDepth = outDepth;
        this.kH = kH;
        this.kW = kW;
        // default SAME padding.
        this.outH = inH;
        this.outW = inW;

    }


    public void feedInput(INDArray input) {
        if (input == null) {
            throw new IllegalArgumentException("Input is null.");
        }
        // verify dimensions
        this.input = input;
    }

    public void feedMockingInput() {
        // minibatch*inDepth*inH*inW
        int[] inputShape = new int[]{miniBatch, inDepth, inH, inW};
        input = Nd4j.valueArrayOf(inputShape, 0.5);
    }

    /**
     * output before activation
     */
    public void preOutput() {

        if (input.rank() != 4) {
            throw new IllegalArgumentException("Input should be 4d.");
        }

        if (input.size(1) != inDepth) {
            throw new IllegalArgumentException("Input Weight depth not match.");
        }

        // from the ND4j comments:
        // "im2col in the required order: want [outW,outH,miniBatch,depthIn,kH,kW]
        // , but need to input [miniBatch,depth,kH,kW,outH,outW] given the current im2col implementation
        // To get this: create an array of the order we want, permute it to the order required by im2col implementation, and then do im2col on that
        // to get old order from required order: permute(0,3,4,5,1,2)
        // Post reshaping: rows are such that minibatch varies slowest, outW fastest as we step through the rows post-reshape"
//        INDArray col = Nd4j.createUninitialized(new int[]{miniBatch, outH, outW, inDepth, kH, kW});
//        INDArray col2 = col.permute(0,3,4,5,1,2); // order: miniBatch, inDepth, kH, kW, outH, outW
//        Convolution.im2col(input, kH, kW, strides[0], strides[1], pad[0], pad[1], true, col2);
//        im2col2d = Shape.newShapeNoCopy(col, new int[]{miniBatch * outH * outW, inDepth * kH * kW}, false);

        if (im2col2d == null) {
            INDArray col = Nd4j.createUninitialized(new int[]{miniBatch, outH, outW, inDepth, kH, kW}, 'c');
            INDArray col2 = col.permute(0, 3, 4, 5, 1, 2);
            Convolution.im2col(input, kH, kW, strides[0], strides[1], pad[0], pad[1], true, col2);
            // Shape im2col to 2d. Due to the permuting above, this should be a zero-copy reshape.
//            im2col2d = col.reshape('c', miniBatch*outH*outW, inDepth*kH*kW);
            im2col2d = Shape.newShapeNoCopy(col, new int[]{miniBatch * outH * outW, inDepth * kH * kW}, false);
        }

        INDArray permutedW = weightsView.permute(3,2,1,0); // not in-place
        INDArray reshapedW = permutedW.reshape('f', kW*kH*inDepth, outDepth);

        z = im2col2d.mmul(reshapedW);
        z.addiRowVector(biasesView);
        permutedW.cleanup();
        z = Shape.newShapeNoCopy(z, new int[]{outW, outH, miniBatch, outDepth}, true);
        z = z.permute(2, 3, 1, 0);


    }

    public INDArray getZ() {
        return z;
    }

    public void  backpropGradient(INDArray epsilon) {
        int[] weightShape = new int[] {outDepth, inDepth, kH, kW};
        INDArray weightGradientView4d = getWeightGradientsView('c', weightShape); // 4d, c order.
        INDArray weightGrad2df = Shape.newShapeNoCopy(weightGradientView4d, new int[]{ outDepth, inDepth*kH*kW}, false).transpose(); // transposed.

//        preOutput();
        INDArray delta = backpropRELU(z, epsilon);  // 4d: nbatch, channel, height, width (NCHW)
        delta = delta.permute(1, 0, 2, 3); // [outDepth, miniBatch, outH, outW]
        // Note: due to the permute in preOut(forward), and the fact that we essentially do a preOut.muli(epsilon), this reshape
        // should be zero-copy; only possible exception being sometimes with the "identity" activation case
        INDArray delta2d = delta.reshape('c', new int[]{outDepth, miniBatch*outH*outW}); //Shape.newShapeNoCopy(delta,new int[]{outDepth,miniBatch*outH*outW},false);

        if (im2col2d == null) {
            INDArray col = Nd4j.createUninitialized(new int[]{miniBatch, outH, outW, inDepth, kH, kW}, 'c');
            INDArray col2 = col.permute(0, 3, 4, 5, 1, 2);
            Convolution.im2col(input, kH, kW, strides[0], strides[1], pad[0], pad[1], true, col2);
            // Shape im2col to 2d. Due to the permuting above, this should be a zero-copy reshape.
//            im2col2d = col.reshape('c', miniBatch*outH*outW, inDepth*kH*kW);
            im2col2d = Shape.newShapeNoCopy(col, new int[]{miniBatch * outH * outW, inDepth * kH * kW}, false);
        }

        // Calculate weight gradients, using cc-> c mmul
        // weightGradients is f order, but this is becuase it's transposed from c order.
        // Here, we are using the fact aht AB = (B^T A^T)^T, output here (post transpose) is in c order, not usual f order.
        Nd4j.gemm(im2col2d, delta2d, weightGrad2df, true, true, 1.0, 0.0); // matrix multiplication

        //flatten 4d weights to 2d, zero copy
        INDArray wPermuted = weightsView.permute(3,2,1,0); // start with c order weights, switch order to f order.
        INDArray w2d = wPermuted.reshape('f', inDepth*kH*kW, outDepth);

        // Calculate epislon for the layer below, in 2d format (note, this is in 'image patch' format before col2im reduction)
        // Note cc -> f mmul here, then reshape to 6d in f order
        epsNext2d = w2d.mmul(delta2d);
//        INDArray eps6d = Shape.newShapeNoCopy(epsNext2d, new int[]{kW, kH, inDepth, outW, outH, miniBatch}, true);
        wPermuted.cleanup();
        // Calculate epsilonNext by doing im2col reduction
        // Currently col2im implementation expects input with order [miniBatch, depth, kH, kW, outH, outW]
        // currently, we have [kH, kW, inDepth, outW, outH, miniBatch]
//        eps6d = eps6d.permute(5, 2, 1, 0, 4, 3); // syi: WHY?
//        INDArray epsNextOrig = Nd4j.create(new int[]{inDepth, miniBatch, inH, inW}, 'c');
//
//        // we are execute col2im in a way that the output array should be used in a stride 1 muli in the layer below ...
//        // same strides as zs/activations
//        INDArray epsNext = epsNextOrig.permute(1,0,2,3); // [miniBatch, inDepth, inH, inW]
//        Convolution.col2im(eps6d, epsNext, strides[0], strides[1], pad[0], pad[1], inH, inW);

        biasGradientsView.assign(delta2d.sum(1));

//        return epsNext2d.ravel();
        // update weight
//        epsNext2d.cleanup();
//        eps6d.cleanup();
//        epsNextOrig.cleanup();
//        epsNext.cleanup();
        delta.cleanup();
    }

    /**
     * a is short for activation
     * L is short for Loss
     * dL/d a(z) = dL/da * dL/dz
     * this function backpropagte the erros through the activation function
     * given input z and epsilon dL/da, where a is the output of a.
     * return dL/dz, calculated from dL/da
     * @param in input before apply the activation,  known as z or preOut.
     * @param epsilon gradients to be backpropagated: dL/da, where L is the loss function.,
     * @return dL/dz
     */
    public INDArray backpropRELU(INDArray in, INDArray epsilon) {
        INDArray dLdz = Nd4j.getExecutioner().execAndReturn(new RectifedLinear(in).derivative()); // // 4d: nbatch, channel, height, width (NCHW)
        dLdz.muli(epsilon);
        return dLdz;  // in-place element wise multiply two NDArrays.
    }

    public void init(INDArray params) {

        int nIn = inDepth*kH*kW;
        int nOut = outDepth;
        paramsFlattened = params;


        // init views from paramsFlattened
        weightsView = paramsFlattened.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn*nOut));
        biasesView = paramsFlattened.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn*nOut, nIn*nOut+nOut));

        // init values in the bias views
        INDArray biasesTmp = Nd4j.valueArrayOf(outDepth, 0.0);
        biasesView.assign(biasesTmp);
        biasesTmp.cleanup();

        // init values in the weight views using method
        // first make it in shape, then flatten it.
        int[] weightShape = new int[]{inDepth*kH*kW, outDepth}; // fanIn, fanOut
        double a = 1.0 / Math.sqrt(weightShape[0]);
        INDArray weightsTmp = Nd4j.rand(weightShape, Nd4j.getDistributions().createUniform(-a, a));
        INDArray weightsFlat = Nd4j.toFlattened('f', weightsTmp); // default order is f.
        weightsView.assign(weightsFlat); // assign values
        weightsFlat.cleanup();

    }

    public INDArray getWeightsView(char order, int[] shape) {
        return weightsView.reshape(order, shape);
    }

    public INDArray getBiasesView() {
        return biasesView;
    }

    /**
     * Receive INDArray gradients from previous layer in backprop
     * @param gradients
     */
    public void setGradientsView(INDArray gradients) {
        this.gradientsFlattened = gradients;
        int nIn = inDepth*kH*kW;
        int nOut = outDepth;
        int[] weightShape = new int[]{outDepth, inDepth, kH, kW};
        weightGradientsView = gradientsFlattened.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, nIn*nOut)).reshape('f', nIn, nOut);
        biasGradientsView = gradientsFlattened.get(NDArrayIndex.point(0), NDArrayIndex.interval(nIn*nOut, nIn*nOut+nOut)); // row vector

    }

    public INDArray getWeightGradientsView(char order, int[] shape) {
        return weightGradientsView.reshape(order, shape);
    }

    public INDArray getBiasGradientsView() {
        return biasGradientsView;
    }


    public static void main(String[] args) {

        String name = "conv1_1";
        int miniBatch = 64;
        int inDepth = 3;
        int inH = 224;
        int inW = 224;
        int outDepth = 64;
        int kH = 3;
        int kW = 3;

        int[] weightShape = new int[] {outDepth, inDepth, kH, kW};
        int nParams = outDepth*inDepth*kH*kW + outDepth;
        VggConvLayer vggConvLayer = new VggConvLayer(name, miniBatch, inDepth, inH, inW, outDepth, kH, kW);


        INDArray gradientsInput = Nd4j.rand(new int[]{nParams});
        INDArray epsilonInput =  Nd4j.rand(new int[]{miniBatch, outDepth, inH, inW});
        INDArray paramsInput = Nd4j.createUninitialized(nParams);

        vggConvLayer.init(paramsInput);

        // mock input
        vggConvLayer.feedMockingInput();

        // forward
        vggConvLayer.preOutput();

        // backward
        vggConvLayer.setGradientsView(gradientsInput);
        vggConvLayer.backpropGradient(epsilonInput);
    }

}
