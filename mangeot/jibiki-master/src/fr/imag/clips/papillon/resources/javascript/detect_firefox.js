function hideGetFirefoxIfAlreadyAvailable() {
    if (isFirefox()) {
        // Hide the get firefox div
        firefoxElement = document.getElementById("firefox"); 
        firefoxElement.style.display = "none";
    }
}

function isFirefox() {
    return ((navigator.userAgent.indexOf("Firefox") != -1) ||
            (navigator.userAgent.indexOf("Iceweasel") != -1));
}