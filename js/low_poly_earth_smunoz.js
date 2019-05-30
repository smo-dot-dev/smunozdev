var camera, scene, renderer, effect
var mesh

var animate = function () {
    requestAnimationFrame(animate)
    mesh.rotation.z += 0.01
    //effect.render( scene, camera )
    renderer.render(scene, camera)
}
var material1 = new THREE.MeshStandardMaterial( {
    opacity: 0.5,
    premultipliedAlpha: true,
    transparent: true
} );
var material2 = new THREE.MeshStandardMaterial( {
    wireframe: true
} );

var setMaterial = function(material){
    mesh.children[2].material[0] = material1
    mesh.children[2].material[1] = material1
    mesh.children[2].material[2] = material1

    mesh.children[3].material[0] = material1
    mesh.children[3].material[1] = material1
    mesh.children[3].material[2] = material1
}

var disableMaterial = function() {
    
}

var container = document.getElementById('canvasthree')

function earth_init() {
    
    const onLoad = (collada) => {
        mesh = collada.scene
        mesh.scale.x = mesh.scale.y = mesh.scale.z = 1
        mesh.updateMatrix()
        //mesh.geometry.material.needsUpdate = true
        scene.add(mesh)
        mesh.rotation.x += 0.6
        console.log("Tierra añadida")
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
    camera = new THREE.PerspectiveCamera(80, window.innerWidth / window.innerHeight, 0.1, 1000)
    camera.position.z = 1.75
    
    //PRECARGA MODELO
    loader = new THREE.ColladaLoader()
    loader.options.convertUpAxis = true
    

    //INICIAMOS RENDER
    renderer = new THREE.WebGLRenderer({
        antialias: false
    })
    renderer.setSize($(container).width(), $(container).height())
    renderer.domElement.style.color = 0xfa8225
    container.appendChild(renderer.domElement)

    //CARGA MODELO Y en callback se llama animate()
    loader.load('low_poly_earth.dae', onError, onProgress, onError)

    // effect = new THREE.AsciiEffect( renderer, ' .:-+*=%@#', { invert: true } )
    // effect.setSize( window.innerWidth, window.innerHeight )
    // effect.domElement.style.color = 'white'
    // effect.domElement.style.backgroundColor = 'black'
    // document.body.appendChild( effect.domElement )
}
