var dim = 128;

(async() => {
    var lrImg = document.getElementById("lr_img"),
        preview = document.getElementById("preview"),
        result = document.getElementById("result"),
        sr_button = document.getElementById("sr_button");

    const model = await tf.loadLayersModel('http://127.0.0.1:5000/model.json');
    //const model = await tf.loadLayersModel('http://127.0.0.1:5000/jsmodels/srcnn_float16_quantization/model.json');
    console.log(model);
        
    lrImg.addEventListener("change", function() {
      changeImage(this);
    });

    sr_button.addEventListener("click", function() {
      srImage();
    });

    function changeImage(input) {
      var reader;

      if (input.files && input.files[0]) {
        reader = new FileReader();

        reader.onload = function(e) {
          preview.setAttribute('src', e.target.result);
        }

        reader.readAsDataURL(input.files[0]);
      }
    }

    function srImage() {
        let modelInput = tf.browser.fromPixels(preview).resizeBilinear([dim, dim]);
        modelInput = modelInput.reshape([1, dim, dim, 3]);
        start = performance.now();
        let prediction = model.predict(modelInput).reshape([dim, dim, 3]);
        time = performance.now() - start;
        console.log(time);
        tf.browser.toPixels(arr, result);
    }
})();