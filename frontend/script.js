const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const snap = document.getElementById("snap");
const errorMsgElement = document.querySelector("span#errorMsg");

const constraints = {
  audio: false,
  video: {
    width: 1280,
    height: 720,
  },
};

// Access webcam
async function init() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    handleSuccess(stream);
  } catch (e) {
    errorMsgElement.innerHTML = `navigator.getUserMedia error:${e.toString()}`;
  }
}

// Success
function handleSuccess(stream) {
  window.stream = stream;
  video.srcObject = stream;
}

function dataURItoBlob(dataURI) {
  // convert base64/URLEncoded data component to raw binary data held in a string
  var byteString;
  if (dataURI.split(',')[0].indexOf('base64') >= 0)
      byteString = atob(dataURI.split(',')[1]);
  else
      byteString = unescape(dataURI.split(',')[1]);

  // separate out the mime component
  var mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];

  // write the bytes of the string to a typed array
  var ia = new Uint8Array(byteString.length);
  for (var i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
  }

  return new Blob([ia], {type:mimeString});
}

// Load init
init();


const getPrediction = async () => {
  var context = canvas.getContext("2d");

  context.drawImage(video, 0, 0, 256, 256);

  var dataURL = canvas.toDataURL("image/jpeg", 0.5);
  var blob = dataURItoBlob(dataURL);

  var formdata = new FormData();
  formdata.append("file", blob, "test1.jpg");

  var requestOptions = {
    method: 'POST',
    body: formdata,
  };
  await fetch("http://localhost:5000/", requestOptions)
  .then(response => response.json())
  .then(result => {
    console.log(result.label)
    document.getElementById("result-text").textContent =  `${result.label}`
    document.getElementById("result-value").textContent =  `${result.percentage.toFixed(2)}%`
  })
  .catch(error => console.log('error', error));
}

const sendPicture = () => {
  setTimeout(() => {
    console.log("enviando")
    getPrediction().then(() => {
      sendPicture()

    })
  }, 100);
}


sendPicture()



