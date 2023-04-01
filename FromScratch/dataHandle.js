function print(val) {
    console.log(val)
}
var values = []
for (let i=0; i<64; i++) {
    values.push(0)
}
function getValues() {
    return values
}
function clearGrid() {
    for (let i=0; i<64; i++) {
        temp = document.getElementById("p"+i)
        temp.style.backgroundColor = light
    }
    values = []
    for (let i=0; i<64; i++) {
        values.push(0)
    }
    print(temp)
    print("Successfully Cleared")
} 
function copyToClipboard() {
    copyText = values
    navigator.clipboard.writeText(copyText);
    // Alert the copied text
    alert("Copied data: " + copyText);
    print("Successfully Copied")
}
function sleep(milliseconds) {
    const date = Date.now();
    let currentDate = null;
    do {
      currentDate = Date.now();
    } while (currentDate - date < milliseconds);
  }



for (let i=0; i<64; i++) {
    temp = document.getElementById("p"+i)
    temp.style.animationDelay = ((i*15)+3000)+"ms";
}


const grid = document.querySelector('#grid-container');
const light = "rgb(203, 210, 219)"
const dark = "darkslategray"
let isDrawing = false;

function draw(e) {
    print(isDrawing)
    if (!isDrawing) return;
    const div = e.target;
    if (div && div.nodeName === 'DIV') {
        if (div.id != "grid-container") {
            idx = String(div.id).slice(1)
            if (div.style.backgroundColor === dark) {
            div.style.backgroundColor = light;
            values[idx] = 0;
            } else {
            div.style.backgroundColor = dark;
            values[idx] = 1;
            }
            print(values)
        }
    }
}

grid.addEventListener('mousedown', (e) => {
    isDrawing = true;
    const div = e.target;
    if (div && div.nodeName === 'DIV') {
        if (div.id != "grid-container") {
            if (div.className == "grid-item") {
                isDrawing = true;
                idx = String(div.id).slice(1)
                if (div.style.backgroundColor === dark) {
                    div.style.backgroundColor = light;
                    values[idx] = 0;
                } else {
                    div.style.backgroundColor = dark;
                    values[idx] = 1;
                }
                print(values)
            }
        }
  }
});

grid.addEventListener('mouseover', draw);

document.addEventListener('mouseup', () => {
  isDrawing = false;
});

function load(data) {
    for (let i=0; i<64; i++) {
        temp = document.getElementById("p"+i)
        if (data[i] == 0) {
            temp.style.backgroundColor = light;
            values[i] = 0;
        } else {
            temp.style.backgroundColor = dark;
            values[i] = 1;
        }
    }
}

let input = document.getElementById("load");

// Execute a function when the user presses a key on the keyboard
input.addEventListener("keypress", function(event) {
  // If the user presses the "Enter" key on the keyboard
  if (event.key === "Enter") {
    // Cancel the default action, if needed
    event.preventDefault();
    // Trigger the button element with a click
    input = JSON.parse("[" + input.value + "]");
    print(input);
    load(input);
    print("Successfully Loaded")
    input = document.getElementById("load");
    input.value = ""
  }
});
         
