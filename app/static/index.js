// let cvsIn = document.getElementById("inputimg");
// let ctxIn = cvsIn.getContext('2d');
// let divOut = document.getElementById("predictdigit");
// let svgGraph = null;
// let mouselbtn = false;
//
// // initilize
// window.onload = function(){
//
//     ctxIn.fillStyle = "white";
//     ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);
//     ctxIn.lineWidth = 7;
//     ctxIn.lineCap = "round";
// }
//
// // add cavas events
// cvsIn.addEventListener("mousedown", function(e) {
//
//     if(e.button == 0){
//         let rect = e.target.getBoundingClientRect();
//         let x = e.clientX - rect.left;
//         let y = e.clientY - rect.top;
//         mouselbtn = true;
//         ctxIn.beginPath();
//         ctxIn.moveTo(x, y);
//     }
//     else if(e.button == 2){
//         onClear();  // right click for clear input
//     }
// });
//
// cvsIn.addEventListener("mouseup", function(e) {
//     if(e.button == 0){
//         mouselbtn = false;
//         onRecognition();
//     }
// });
// cvsIn.addEventListener("mousemove", function(e) {
//     let rect = e.target.getBoundingClientRect();
//     let x = e.clientX - rect.left;
//     let y = e.clientY - rect.top;
//     if(mouselbtn){
//         ctxIn.lineTo(x, y);
//         ctxIn.stroke();
//     }
// });
//
// cvsIn.addEventListener("touchstart", function(e) {
//     // for touch device
//     if (e.targetTouches.length == 1) {
//         let rect = e.target.getBoundingClientRect();
//         let touch = e.targetTouches[0];
//         let x = touch.clientX - rect.left;
//         let y = touch.clientY - rect.top;
//         ctxIn.beginPath();
//         ctxIn.moveTo(x, y);
//     }
// });
//
// cvsIn.addEventListener("touchmove", function(e) {
//     // for touch device
//     if (e.targetTouches.length == 1) {
//         let rect = e.target.getBoundingClientRect();
//         let touch = e.targetTouches[0];
//         let x = touch.clientX - rect.left;
//         let y = touch.clientY - rect.top;
//         ctxIn.lineTo(x, y);
//         ctxIn.stroke();
//         e.preventDefault();
//     }
// });
//
// cvsIn.addEventListener("touchend", function(e) {
//     // for touch device
//     onRecognition();
// });
//
// // prevent display the contextmenu
// cvsIn.addEventListener('contextmenu', function(e) {
//     e.preventDefault();
// });
//
// document.getElementById("clearbtn").onclick = onClear;
//
// function onClear(){
//     mouselbtn = false;
//     ctxIn.fillStyle = "white";
//     ctxIn.fillRect(0, 0, cvsIn.width, cvsIn.height);
//     ctxIn.fillStyle = "black";
// }
//
// // post data to server for recognition
// function onRecognition() {
//     console.time("predict");
//
//     $.ajax({
//             url: './predict',
//             type:'POST',
//             data : {img : cvsIn.toDataURL("image/png").replace('data:image/png;base64,','') },
//
//         }).done(function(data) {
//
//             showResult(JSON.parse(data))
//
//         }).fail(function(XMLHttpRequest, textStatus, errorThrown) {
//             console.log(XMLHttpRequest);
//             alert("error");
//         })
//
//     console.timeEnd("time");
// }
//
//
// function showResult(resultJson){
//
//     // show predict digit
//     divOut.textContent = resultJson.prediction;
//
//     // show probability
//     document.getElementById("probStr").innerHTML =
//         "Probability : " + resultJson.probability.toFixed(2) + "%";
//
// }
//predict
//
// function drawImgToCanvas(canvasId, b64Img){
//     let canvas = document.getElementById(canvasId);
//     let ctx = canvas.getContext('2d');
//     let img = new Image();
//     img.src = "data:image/png;base64," + b64Img;
//     img.onload = function(){
//         ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, canvas.width, canvas.height);
//     }
// }


//jQuery 라이브러리 이용


function dragOver(e) {
    e.stopPropagation();
    e.preventDefault();
    if (e.type == "dragover") {
        $(e.target).css({
            "background-color": "black",
            "outline-offset": "-20px"
        });
    } else {
        $(e.target).css({
            "background-color": "gray",
            "outline-offset": "-10px"
        });
    }
}

function uploadFiles(e) {
    e.stopPropagation();
    e.preventDefault();
    dragOver(e);

    e.dataTransfer = e.originalEvent.dataTransfer;
    var files = e.target.files || e.dataTransfer.files;
    if (files.length > 1) {
        alert('하나만 올려라.');
        return;
    }
    if (files[0].type.match(/image.*/)) {
        const blob = window.URL.createObjectURL(files[0]);
        $(e.target).css({
            "background-image": "url(" + blob + ")",
            "outline": "none",
            "background-size": "100% 100%"
        });
        const formData = new FormData(document.getElementById('myform'));
        formData.append('image', files[0]);
        fetch('/predict', {
            method: 'POST',
            body:formData
        })
    } else {
        alert('이미지가 아닙니다.');
        return;
    }

}

$('.content1')
    .on("dragover", dragOver)
    .on("dragleave", dragOver)
    .on("drop", uploadFiles);
