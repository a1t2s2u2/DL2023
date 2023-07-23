const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

canvas.width = 2000; canvas.height = 2000;
const SHAPE = [150, 150];

const d = 1;

class Model {
  constructor() {
    this.neurons = Tensor.random([...SHAPE, 3]);
    this.weights = Tensor.random([...SHAPE, 2*d+1, 2*d+1, 3]);
    setInterval(this.loop.bind(this), 200);
  }
  loop() {
    // 値の送信
    let copy = JSON.parse(JSON.stringify(this.neurons));
    copy.forEach((line, y) => {
      if(d <= y && y < copy.length-d) {
        line.forEach((neuron, x) => {
          if(d <= x && x < line.length-d) {
            const weights = this.weights[y][x];
            for(var j=-d;j<=d;++j) {
              for(var i=-d;i<=d;++i) {
                this.neurons[y+j][x+i] = Tensor.calc(
                  this.neurons[y+j][x+i],
                  Tensor.calc(weights[d+j][d+i], neuron, "mul"),
                  "add"
                );
              }
            }
          }
        });
      }
    });

    // 正規化
    this.neurons = Tensor.normalize(this.neurons);

    this.draw();
  }
  draw() {
    const a = Tensor.calc([canvas.width, canvas.height], SHAPE, "div");
    this.neurons.forEach((line, y) => {
      line.forEach((neuron, x) => {
        const rgb = Tensor.calc(255, neuron, "mul");
        ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
        ctx.fillRect(x*a[0], y*a[1], a[0], a[1]);
      });
    });
  }
}

let model = new Model();