package net.yishanhe.deepandroidvgg;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.annotation.NonNull;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Toast;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;

import java.io.File;
import java.io.IOException;

public class MainActivity extends AppCompatActivity implements SharedPreferences.OnSharedPreferenceChangeListener {

    public static final String  TAG = "DeepAndroidVGG";
    private SharedPreferences prefs;

    // Permissions
    private final static int REQUEST_PERMISSIONS = 1;
    private final static int INTENT_CAMERA = 1;
    private final static String[] PERMISSIONS = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.READ_EXTERNAL_STORAGE,
            Manifest.permission.INTERNET, Manifest.permission.ACCESS_NETWORK_STATE};

    // TODO build better image injection.
    private static final String IMAGE_FOLDER = "/sdcard/lfw-224/train_data";

    private int miniBatch = 1;
    private int iterations = 1;

    VggConvLayer vggConvLayer;
    VggConvLayer vggConvLayer1;

    INDArray gradientsInput;
    INDArray gradientsInput1;
    INDArray epsilonInput;
    INDArray paramsInput;
    INDArray paramsInput1;
    Activity activity;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        activity = this;
        if (!hasPermissions(MainActivity.this, PERMISSIONS)) {
            ActivityCompat.requestPermissions(MainActivity.this, PERMISSIONS, REQUEST_PERMISSIONS);
        }

        prefs = PreferenceManager.getDefaultSharedPreferences(this);
        miniBatch = Integer.parseInt(prefs.getString("minibatch", "1"));
        iterations = Integer.parseInt(prefs.getString("iterations", "1"));






        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                // Do action here.
                new AsyncTask<Void, Void, Boolean>() {
                    @Override
                    protected Boolean doInBackground(Void... params) {

//                        conv1x1();
                        conv1x2();
                        activity.runOnUiThread(new Runnable() {
                            public void run() {
                                Toast.makeText(activity, "Hello", Toast.LENGTH_SHORT).show();
                            }
                        });
                        return true;
                    }
                }.execute();
            }
        });
    }

    public void conv1x1() {

        String name = "conv1_1";
        int inDepth = 3;
        int inH = 224;
        int inW = 224;
        int outDepth = 64;
        int kH = 3;
        int kW = 3;

        int nParams = outDepth*inDepth*kH*kW + outDepth;
        vggConvLayer = new VggConvLayer(name, miniBatch, inDepth, inH, inW, outDepth, kH, kW);
        paramsInput = Nd4j.createUninitialized(nParams);
        vggConvLayer.init(paramsInput);


        gradientsInput = Nd4j.rand(new int[]{nParams});
        epsilonInput =  Nd4j.rand(new int[]{miniBatch, outDepth, inH, inW});




        long sTime;
        long eTime;

        long batchLoadingTime = 0;
        long forwardTime = 0;
        long backwardTime = 0;

        Log.d(TAG, "Training Starts...");
        Log.d(TAG, "Minibatch = " + miniBatch);
        Log.d(TAG, "Iterations = " + iterations);

        for (int i = 0; i < iterations; i++) {
            Log.d(TAG, "Iteration "+(i+1));

            sTime = System.currentTimeMillis();
            INDArray input = loadBatchedBitmaps(miniBatch);
            vggConvLayer.feedInput(input);
            eTime = System.currentTimeMillis();
            batchLoadingTime += (eTime-sTime);
            Log.d(TAG, "batch loading time: "+((eTime-sTime)/1000.0)+"s");


            sTime = System.currentTimeMillis();
            vggConvLayer.preOutput();
            eTime = System.currentTimeMillis();
            forwardTime += (eTime-sTime);
            Log.d(TAG, "forward time: "+((eTime-sTime)/1000.0)+"s");

//            System.out.println("Data Out: " + vggConvLayer.getZ().shapeInfoToString());
//            System.out.println("Data In" + epsilonInput.shapeInfoToString());
//            try {
//                System.out.println("Data Out: " + (Nd4j.toByteArray(vggConvLayer.getZ()).length/1000000.0) + "MB.");
//                System.out.println("Data In: " + (Nd4j.toByteArray(epsilonInput).length/1000000.0) + "MB.");
//            } catch (IOException e) {
//                e.printStackTrace();
//            }


            sTime = System.currentTimeMillis();
            vggConvLayer.setGradientsView(gradientsInput);
            vggConvLayer.backpropGradient(epsilonInput);
            eTime = System.currentTimeMillis();
            backwardTime += (eTime-sTime);
            Log.d(TAG, "backward time: "+((eTime-sTime)/1000.0)+"s");

        }


        Log.d(TAG, "Training Ends...");
        Log.d(TAG, "Stats:");
        Log.d(TAG, "Average batch loading time:" + (batchLoadingTime/iterations/1000.0) + "s");
        Log.d(TAG, "Average forward time:" + (forwardTime/iterations/1000.0) + "s");
        Log.d(TAG, "Average backward time:" + (backwardTime/iterations/1000.0) + "s");

    }
    public void conv1x2() {

        String name = "conv1_2";
        int inDepth = 3;
        int inH = 224;
        int inW = 224;
        int outDepth = 64;
        int kH = 3;
        int kW = 3;

        int nParams = outDepth*inDepth*kH*kW + outDepth;
        vggConvLayer = new VggConvLayer(name, miniBatch, inDepth, inH, inW, outDepth, kH, kW);
        paramsInput = Nd4j.createUninitialized(nParams);
        gradientsInput = Nd4j.rand(new int[]{nParams});
        vggConvLayer.init(paramsInput);

        int nParams1 = outDepth*outDepth*kH*kW + outDepth;
        vggConvLayer1 = new VggConvLayer(name, miniBatch, outDepth, inH, inW, outDepth, kH, kW);
        paramsInput1 = Nd4j.createUninitialized(nParams1);
        gradientsInput1 = Nd4j.rand(new int[]{nParams1});
        vggConvLayer1.init(paramsInput1);


        epsilonInput =  Nd4j.rand(new int[]{miniBatch, outDepth, inH, inW});



        long sTime;
        long eTime;

        long batchLoadingTime = 0;
        long forwardTime = 0;
        long backwardTime = 0;
        long forwardTime1 = 0;
        long backwardTime1 = 0;

        Log.d(TAG, "Training Starts...");
        Log.d(TAG, "Minibatch = " + miniBatch);
        Log.d(TAG, "Iterations = " + iterations);

        for (int i = 0; i < iterations; i++) {
            Log.d(TAG, "Iteration "+(i+1));

            sTime = System.currentTimeMillis();
            INDArray input = loadBatchedBitmaps(miniBatch);
            vggConvLayer.feedInput(input);
            eTime = System.currentTimeMillis();
            batchLoadingTime += (eTime-sTime);
            Log.d(TAG, "batch loading time: "+((eTime-sTime)/1000.0)+"s");


            Log.d(TAG, "Layer 1");

            sTime = System.currentTimeMillis();
            vggConvLayer.preOutput();
            eTime = System.currentTimeMillis();
            forwardTime += (eTime-sTime);
            Log.d(TAG, "forward time: "+((eTime-sTime)/1000.0)+"s");



            Log.d(TAG, "Layer 2");

            sTime = System.currentTimeMillis();
            vggConvLayer1.feedInput(vggConvLayer.getZ());
            vggConvLayer1.preOutput();
            eTime = System.currentTimeMillis();
            forwardTime1 += (eTime-sTime);
            Log.d(TAG, "forward time: "+((eTime-sTime)/1000.0)+"s");


//            System.out.println("Data Out: " + vggConvLayer1.getZ().shapeInfoToString());
//            System.out.println("Data In" + epsilonInput.shapeInfoToString());
//            try {
//                System.out.println("Data Out: " + (Nd4j.toByteArray(vggConvLayer.getZ()).length/1000000.0) + "MB.");
//                System.out.println("Data In: " + (Nd4j.toByteArray(epsilonInput).length/1000000.0) + "MB.");
//            } catch (IOException e) {
//                e.printStackTrace();
//            }

            sTime = System.currentTimeMillis();
            vggConvLayer1.setGradientsView(gradientsInput1);
            vggConvLayer1.backpropGradient(epsilonInput);
            eTime = System.currentTimeMillis();
            backwardTime1 += (eTime-sTime);
            Log.d(TAG, "backward time: "+((eTime-sTime)/1000.0)+"s");

//                            vggConvLayer1.cleanup();

            Log.d(TAG, "Layer 1");

            sTime = System.currentTimeMillis();
            vggConvLayer.setGradientsView(gradientsInput);
            vggConvLayer.backpropGradient(epsilonInput);
            eTime = System.currentTimeMillis();
            backwardTime += (eTime-sTime);
            Log.d(TAG, "backward time: "+((eTime-sTime)/1000.0)+"s");


//                            input.cleanup();
//                            vggConvLayer.cleanup();
        }


        Log.d(TAG, "Training Ends...");
        Log.d(TAG, "Stats:");
        Log.d(TAG, "Average batch loading time:" + (batchLoadingTime/iterations/1000.0) + "s");
        Log.d(TAG, "Average forward time:" + (forwardTime/iterations/1000.0) + "s");
        Log.d(TAG, "Average forward1 time:" + (forwardTime1/iterations/1000.0) + "s");
        Log.d(TAG, "Average backward1 time:" + (backwardTime1/iterations/1000.0) + "s");
        Log.d(TAG, "Average backward time:" + (backwardTime/iterations/1000.0) + "s");
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            startActivity(new Intent(this, SettingActivity.class));
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case REQUEST_PERMISSIONS:
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Permission Granted", Toast.LENGTH_SHORT).show();
                    recreate();
                }
                else {
                    Toast.makeText(this, "Fail to get permission.", Toast.LENGTH_SHORT).show();
                    finish();
                }
        }
    }


    public static boolean hasPermissions(Context context, String[] permissions) {
        if (context!=null) {
            for (String permission :
                    permissions) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }

//    public static INDArray loadSingleBitmap() {
//        int inputSize = 224;
//        Bitmap bitmap = BitmapFactory.decodeFile(IMAGE_FOLDER+"/Alejandro_Toledo/Alejandro_Toledo_0002.png");
//        int[] intImageValues = new int[inputSize*inputSize];
//        float[] floatImageValues = new float[inputSize*inputSize*3];
//        int imageMean = 113 ;
//        float imageStd = 1;
//        bitmap.getPixels(intImageValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
//        bitmap.recycle();
//        for (int i = 0; i < intImageValues.length; ++i) {
//            final int val = intImageValues[i];
//            floatImageValues[i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
//            floatImageValues[i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
//            floatImageValues[i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
//        }
//        INDArray input = Nd4j.create(floatImageValues, new int[]{1,3,inputSize, inputSize}, 'c');
//        return input;
//    }

    public static INDArray loadBatchedBitmaps(int miniBatch) {
        int batchCount = 0;
        int inputSize = 224;
        int[] intImageValues = new int[inputSize*inputSize];
        int imageMean = 113 ;
        float imageStd = 1;

        double[][] doubleBatchedImageValues = new double[miniBatch][inputSize*inputSize*3];
        for( File labeledFolder: new File(IMAGE_FOLDER).listFiles()) {
            if (!labeledFolder.isDirectory()) continue;

            if (batchCount<miniBatch) {
                for(File imageFile: labeledFolder.listFiles()) {
                    if (!imageFile.isFile()) continue;

                    if(batchCount<miniBatch) {
                        Bitmap bitmap = BitmapFactory.decodeFile(imageFile.getAbsolutePath());
                        bitmap.getPixels(intImageValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

                        for (int i = 0; i < intImageValues.length; ++i) {
                            final int val = intImageValues[i];
                            doubleBatchedImageValues[batchCount][i * 3 + 0] = (((val >> 16) & 0xFF) - imageMean) / imageStd;
                            doubleBatchedImageValues[batchCount][i * 3 + 1] = (((val >> 8) & 0xFF) - imageMean) / imageStd;
                            doubleBatchedImageValues[batchCount][i * 3 + 2] = ((val & 0xFF) - imageMean) / imageStd;
                        }
                        bitmap.recycle();
                        batchCount++;
                    } else {
                        break;
                    }
                }
            } else {
                break;
            }
        }

        INDArray input = Nd4j.create(ArrayUtil.flattenDoubleArray(doubleBatchedImageValues), new int[]{miniBatch,3,inputSize, inputSize}, 'c');
//        System.out.println(input.shapeInfoToString());
        return input;
    }


    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        Log.d(TAG, "onSharedPreferenceChanged: "+key);
        if (key.equals("minibatch")) {
            miniBatch = Integer.parseInt(prefs.getString("minibatch", "1"));
        }

        if (key.equals("iterations")) {
            iterations = Integer.parseInt(prefs.getString("iterations", "1"));
        }
    }
}
