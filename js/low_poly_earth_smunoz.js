var camera, scene, renderer, effect
var mesh


var animate = function () {
    requestAnimationFrame(animate)
    mesh.rotation.z += 0.016
    //effect.render( scene, camera )
    renderer.render(scene, camera)
}
var material1 = new THREE.MeshStandardMaterial( {
    opacity: 0.5,
    premultipliedAlpha: true,
    transparent: true
} );

var material2a, material2b

var m1enabled = false

var toggleMaterial1 = function(){
    if (!m1enabled) {
        mesh.children[2].material[0] = material1
        mesh.children[3].material[0] = material1
        m1enabled = true
        $("#linkedin").addClass("invertir")
    }else{
        mesh.children[2].material[0] = material2a
        mesh.children[3].material[0] = material2b
        m1enabled = false
        $("#linkedin").removeClass("invertir")
    }
}

var toggleMaterial2 = function(){
    if (mesh.children[2].material[0].wireframe) {
        mesh.children[2].material[0].wireframe = false
        mesh.children[2].material[1].wireframe = false
        mesh.children[2].material[2].wireframe = false
        $("#github").removeClass("invertir")
    }else{
        mesh.children[2].material[0].wireframe = true
        mesh.children[2].material[1].wireframe = true
        mesh.children[2].material[2].wireframe = true
        
        $("#github").addClass("invertir")
    }
    
}


function earth_init() {
    
    const onLoad = (collada) => {
        mesh = collada.scene
        mesh.scale.x = mesh.scale.y = mesh.scale.z = 1
        mesh.updateMatrix()
        //mesh.geometry.material.needsUpdate = true
        scene.add(mesh)
        mesh.rotation.x += 0.6
        console.log("Tierra añadida")
        material2a= mesh.children[2].material[0]
        material2b= mesh.children[3].material[0]
        animate()
    }

    const onProgress = () => {}

    const onError = (errorMessage) => {
        console.log(errorMessage)
        var geometry = new THREE.BoxGeometry(1.3,1.3,1.3)
        var material = new THREE.MeshNormalMaterial()
        mesh = new THREE.Mesh(geometry, material)
        scene.add(mesh)
        mesh.rotation.x += 90
        console.log("\n...Creando cubo aburrido")
        animate()
    }

    //ESCENA
    scene = new THREE.Scene()
    scene.background = new THREE.Color( 0xfa8225 ) //tf : 0xff9200 bg: 0xfa8225
    scene.add( new THREE.AmbientLight( 0xffffff, 0.5))
    
    //CÁMARA
    camera = new THREE.PerspectiveCamera(80, 1, 0.1, 1000)
    camera.position.z = 1.62
    
    
    //PRECARGA MODELO
    loader = new THREE.ColladaLoader()
    loader.options.convertUpAxis = true
    

    //INICIAMOS RENDER
    renderer = new THREE.WebGLRenderer({
        antialias: false
    })
    var container = document.getElementById('canvasthree')
    renderer.setSize(420,420) //Uso valores literales pa no usar jquery
    renderer.domElement.style.color = 0xfa8225
    container.appendChild(renderer.domElement)

    //CARGA MODELO Y en callback se llama animate()
    loader.load('./js/low_poly_earth.dae', onLoad, onProgress, onError)

    // effect = new THREE.AsciiEffect( renderer, ' .:-+*=%@#', { invert: true } )
    // effect.setSize( window.innerWidth, window.innerHeight )
    // effect.domElement.style.color = 'white'
    // effect.domElement.style.backgroundColor = 'black'
    // document.body.appendChild( effect.domElement )
}
