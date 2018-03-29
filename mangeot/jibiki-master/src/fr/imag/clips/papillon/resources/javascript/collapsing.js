/*
 * Collapsing the content of lexal entries.
 */

// The node that is passed is either a contribution or a translation class
function entryShowHide(node) {
    var tr = node.firstChild;
    var eh = /entry\_header/i;
    var ei = /entry\_information/i;
    var divs = node.getElementsByTagName("div");
    var i;

    for (i = 0; i < divs.length; i++) {
        if (divs[i].className == 'collapsable') toggleDisplay(divs[i]);
    }

//    while (tr) {
//        if (tr.nodeType == 1 && // ELEMENT_NODE
//            tr.tagName.match(/^div$/i)) {
//            classString = tr.className;
//
//            if (! (
//                    classString.match(eh) || classString.match(ei) )) {
//                toggleDisplay(tr);
//            }
//        }
//        tr = tr.nextSibling;
//    }
}

function toggleDisplay(elem) {
    var vis = elem.style;
    // if the style.display value is blank we try to figure it out here
    if (vis.display == '' && elem.offsetWidth != undefined && elem.offsetHeight != undefined) {
        vis.display = (elem.offsetWidth != 0 && elem.offsetHeight != 0) ? 'block' : 'none';
    }
    vis.display = (vis.display == '' || vis.display == 'block') ? 'none' : 'block';
}