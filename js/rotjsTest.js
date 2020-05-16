var smunozterm = {
  display: null,
  simplex: null,
  fontSize: 14,
  width: 80,
  height: 40,
  time: 0,
  timeScale: 0.025,
  ready: false,
  word: 'LinkedIn  ', // 'Welcome to smunoz.dev!
  wc: -1,
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
        fontFamily: "monospace",
        fg: "white",
        bg: "#888",
      });
    }
  },
  init: function () {
    this.display = new ROT.Display();
    this.simplex = new SimplexNoise();
    if (this.display != null && this.simplex != null) {
      this.ready = true;
    }
    this.updateDisplayConfig(this.width,this.height, this.fontSize);

  },
  draw: function () {
    if (this.ready) {
      for (var j = 0; j < this.height; j++) {
        for (var i = 0; i < this.width; i++) {
          var val = this.simplex.noise3D(i / 28, j / 28, this.time)*255;
          if (val < -15){
            this.display.draw(i, j, ' ','', "rgb(" + (-val) + "," + (-val)  + "," + (-val) + ")");
          }
          else{
            this.display.draw(i, j, this.word[this.wc],'', "rgb(" + 0 + "," + 0 + "," + val + ")");
          }
          this.wc += 1;
          if (this.wc == this.word.length){
            this.wc = 0;
          }
        }
        
      }
      this.time += this.timeScale;
    }
  }

};


//Entry-point
$(function() {
  smunozterm.init();
  document.getElementById("smunozterm").appendChild(smunozterm.display.getContainer());
  setInterval(function () {smunozterm.draw();}, 42); //42ms ~= 24 fps
});