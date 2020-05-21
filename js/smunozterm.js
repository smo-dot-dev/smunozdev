var smunozterm = {
  display: null,
  simplex: null,
  fontSize: 16,
  width: 80,
  height: 25,
  time: 0,
  timeScale: 0.022,
  word: 'smunoz.dev......', // 'Welcome to smunoz.dev!
  wc: 0,
  updateDisplayConfig: function (w, h, fontSize) {
    this.width = w;
    this.height = h;
    this.fontSize = fontSize;
    if (this.display != null) {
      this.display.setOptions({
        width: w, //~~(window.innerWidth/this.fontSize)/2,
        height: h, //~~(window.window.innerHeight/this.fontSize),
        fontSize: fontSize,
        forceSquareRatio: true,
        fontFamily: "deltaray",
        fg: "#CECEC6",
        bg: "#6b635a",
      });
    }
  },
  init: function () {
    this.display = new ROT.Display();
    this.simplex = new SimplexNoise();
    this.updateDisplayConfig(this.width, this.height, this.fontSize);
  },
  draw: function () {
    for (var j = 0; j < this.height; j++) {
      for (var i = 0; i < this.width; i++) {
        var val = this.simplex.noise3D(i / 30, j / 30, this.time) * 255;
        if (val < -15) {
          this.display.draw(i, j, ' ');
        } else {
          val +=70;
          this.display.draw(i, j, this.word[this.wc], "rgb(" + val + ','+ val + ','+ val + ')');
        }
        this.wc += 1;
        if (this.wc == this.word.length) {
          this.wc = 0;
        }
      }
    }
    this.time += this.timeScale;
  },
  changeWord: function (word, scl) {
    this.word = word
    this.wc = 0
  }
};

var defaultWord = "smunoz.dev......";


//Entry-point jQuery
$(function () {
  smunozterm.init();
  document.getElementById("smunozterm").appendChild(smunozterm.display.getContainer());
  setInterval(function () {
    smunozterm.draw();
  }, 34); //33ms ~= 30 fps
});