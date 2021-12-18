package com.example.imagesuperresolution;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;

import com.example.imagesuperresolution.ml.Edsr;
import com.example.imagesuperresolution.ml.EdsrDynamicQuantization;
import com.example.imagesuperresolution.ml.EdsrFloat16Quantization;
import com.example.imagesuperresolution.ml.SrcnnDynamicQuantization;
import com.example.imagesuperresolution.ml.SrcnnFloat16Quantization;
import com.example.imagesuperresolution.ml.WdsrDynamicQuantization;
import com.example.imagesuperresolution.ml.WdsrFloat16Quantization;

import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button upsampleImage = findViewById(R.id.button);
        upsampleImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ImageView image = findViewById(R.id.imageView);
                Bitmap bitmap = ((BitmapDrawable) image.getDrawable()).getBitmap();
                try {
                    WdsrDynamicQuantization model = WdsrDynamicQuantization.newInstance(getApplicationContext());
                    TensorImage tfImage = new TensorImage();
                    tfImage.load(bitmap);
                    long start = System.nanoTime();
                    WdsrDynamicQuantization.Outputs outputs = model.process(tfImage.getTensorBuffer());
                    long finish = System.nanoTime();
                    TensorBuffer outputFeatures = outputs.getOutputFeature0AsTensorBuffer();
                    long timeElapsed = finish - start;
                    System.out.println("Model Inference Time: " + timeElapsed + " ns");
                    System.out.println(timeElapsed);
                    model.close();
                } catch (IOException e) {
                    // TODO Handle the exception
                }
            }
        });
    }
}