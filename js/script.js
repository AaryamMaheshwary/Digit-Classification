function load_data(data) {
  w1 = math.matrix(data.w1);
  w2 = math.matrix(data.w2);
  b1 = math.matrix(data.b1);
  b2 = math.matrix(data.b2);
}

function reset() {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "black";
  ctx.lineWidth = 20;
  ctx.lineCap = "round";
  update_prediction("--", "--");
}

function get_canvas_data() {
  let X = [];
  let pixel_size = canvas.height / 28;
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

function forward_prop(X) {
  let z1 = math.add(math.multiply(w1, X), b1);
  let a1 = z1.map(function (value) {
    return math.max(value, 0);
  });
  let z2 = math.add(math.multiply(w2, a1), b2);
  let a2 = math.divide(math.exp(z2), math.sum(math.exp(z2)));
  return a2;
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

function classify() {
  let X = get_canvas_data();
  let a2 = forward_prop(X);
  let y = get_prediction(a2);
  update_prediction(y.pred, y.conf);
}

function start_draw(event) {
  document.addEventListener("mousemove", draw);
  update_coords(event);
}

function stop_draw() {
  document.removeEventListener("mousemove", draw);
}

function update_coords(event) {
  coords.x = event.clientX - canvas.offsetLeft;
  coords.y = event.clientY - canvas.offsetTop;
}

function make_stroke() {
  let pixel_size = canvas.height / 28;
  let x = Math.floor(coords.x / pixel_size) * pixel_size;
  let y = Math.floor(coords.y / pixel_size) * pixel_size;
  ctx.fillRect(x, y, pixel_size, pixel_size);
  ctx.globalAlpha = 0.75;
  ctx.fillRect(x - pixel_size, y, pixel_size, pixel_size);
  ctx.fillRect(x + pixel_size, y, pixel_size, pixel_size);
  ctx.fillRect(x, y - pixel_size, pixel_size, pixel_size);
  ctx.fillRect(x, y + pixel_size, pixel_size, pixel_size);
  ctx.globalAlpha = 1;
}

function draw(event) {
  ctx.beginPath();
  update_coords(event);
  make_stroke();
  ctx.closePath();
  classify()
}

function update_prediction(pred, conf) {
  document.getElementById("prediction").innerHTML = pred;
  document.getElementById("confidence").innerHTML = conf;
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
var coords = {
  x: 0,
  y: 0,
};
reset();

// Start listening for mouse clicks to draw on the canvas
document.addEventListener("mousedown", start_draw);
document.addEventListener("mouseup", stop_draw);
