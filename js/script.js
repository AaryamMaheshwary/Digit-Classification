function reset() {
  ctx.globalAlpha = 1;
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "black";
  update_prediction_label("--", "--");
}

function update_prediction_label(pred, conf) {
  document.getElementById("prediction").innerHTML = pred;
  document.getElementById("confidence").innerHTML = conf;
}

function get_prediction(a2) {
  let conf = 0, pred;
  a2.forEach(function (value, index) {
    if (value > conf) {
      conf = value;
      pred = index[0];
    }
  });
  conf = Math.round(conf * 1000) / 10;
  return { pred, conf };
}

function forward_prop(X) {
  let z1 = math.add(math.multiply(w1, X), b1);
  let a1 = z1.map(function (value) {
    return math.max(value, 0);
  });
  let z2 = math.add(math.multiply(w2, a1), b2);
  let a2 = math.divide(math.exp(z2), math.sum(math.exp(z2)));
  return a2;
}

function get_canvas_data() {
  let X = [];
  for (let i = 0; i < 28; i++) {
    for (let j = 0; j < 28; j++) {
      let x = j * pixel_size + pixel_size / 2;
      let y = i * pixel_size + pixel_size / 2;
      let pixel = ctx.getImageData(x, y, 1, 1);
      let shade = 1 - pixel.data[0] / 255;
      X.push([shade]);
    }
  }
  return X;
}

function classify() {
  let X = get_canvas_data();
  let a2 = forward_prop(X);
  let y = get_prediction(a2);
  update_prediction_label(y.pred, y.conf);
}

function fill_pixel(x, y, alpha) {
  ctx.globalAlpha = alpha;
  ctx.fillRect(x, y, pixel_size, pixel_size);
}

function make_stroke(x, y) {
  x = Math.floor(x / pixel_size) * pixel_size;
  y = Math.floor(y / pixel_size) * pixel_size;
  fill_pixel(x, y, 1);
  fill_pixel(x - pixel_size, y, 0.75);
  fill_pixel(x + pixel_size, y, 0.75);
  fill_pixel(x, y - pixel_size, 0.75);
  fill_pixel(x, y + pixel_size, 0.75);
}

function draw(event) {
  let x = event.clientX - canvas.offsetLeft;
  let y = event.clientY - canvas.offsetTop;
  make_stroke(x, y);
}

function start_draw() {
  document.addEventListener("mousemove", draw);
}

function stop_draw() {
  document.removeEventListener("mousemove", draw);
}

function load_data(data) {
  w1 = math.matrix(data.w1);
  w2 = math.matrix(data.w2);
  b1 = math.matrix(data.b1);
  b2 = math.matrix(data.b2);
}

// Load the model parameters
var w1, w2, b1, b2;
fetch("model/model_params.json")
  .then((response) => response.json())
  .then((data) => load_data(data))
  .catch((error) => console.log(error));

// Setup the canvas
var canvas = document.getElementById("canvas");
var ctx = canvas.getContext("2d");
const pixel_size = canvas.height / 28;
reset();

// Start listening for mouse clicks to draw on the canvas
document.addEventListener("mousedown", start_draw);
document.addEventListener("mouseup", stop_draw);
