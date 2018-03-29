/**
 * AutoComplete Field - JavaScript Code
 *
 * This is a sample source code provided by fromvega.
 * Search for the complete article at http://www.fromvega.com
 *
 * Enjoy!
 *
 * Adapted for jibiki use by Gilles Sérasset
 *
 * @author fromvega
 *
 */

// global variables
var acListTotal   =  0;
var acListCurrent = -1;
var acDelay		  = 500;
var acURL		  = null;
var acLangParam	  = null;
var acHeadwordParam		  = null;
var acSearchId	  = null;
var acLangId	  = null;
var acResultsId	  = null;
var acSearchField = null;
var acLangField   = null;
var acResultsDiv  = null;

function setAutoComplete(search_field_id, lang_field_id, results_id, url, hw, lang ){

	// initialize vars
	acSearchId  = "#" + search_field_id;
    acLangId  = "#" + lang_field_id;
	acResultsId = "#" + results_id;
	acURL 		= url;
    acLangParam   = lang;
    acHeadwordParam     = hw;

    // create the results div
	$("body").append('<div id="' + results_id + '"></div>');

	// register mostly used vars
	acSearchField	= $(acSearchId);
    acLangField	    = $(acLangId);
	acResultsDiv	= $(acResultsId);

	// reposition div
	repositionResultsDiv();
	
	// on blur listener
	acSearchField.blur(function(){ setTimeout("clearAutoComplete()", 200) });

    // disallow firefox to autocomplete the field himself
    document.getElementById(search_field_id).setAttribute("autocomplete", "off");
    
    // on key up listener
	acSearchField.keyup(function (e) {

		// get keyCode (window.event is for IE)
		var keyCode = e.keyCode || window.event.keyCode;
		var lastVal = acSearchField.val();

		// check an treat up and down arrows
		if(updownArrow(keyCode)){
			return;
		}

		// check for an ENTER or ESC
		if(keyCode == 13 || keyCode == 27){
			clearAutoComplete();
			return;
		}

		// if is text, call with delay
		setTimeout(function () {autoComplete(lastVal)}, acDelay);
	});
}

// treat the auto-complete action (delayed function)
function autoComplete(lastValue)
{
	// get the field value
	var part = acSearchField.val();
    var lg = acLangField.val();

	// if it's empty clear the resuts box and return
	if(part == ''){
		clearAutoComplete();
		return;
	}

	// if it's equal the value from the time of the call, allow
	if(lastValue != part){
		return;
	}

	// get remote data as JSON
	$.getJSON(acURL + "?" + acLangParam + lg + "&" + acHeadwordParam + part, function(json){

		// get the total of results
		var ansLength = acListTotal = json.length;

		// if there are results populate the results div
		if(ansLength > 0){

			var newData = '';

			// create a div for each result
			for(i=0; i < ansLength; i++) {
				newData += '<div class="unselected">' + json[i] + '</div>';
			}

			// update the results div
			acResultsDiv.html(newData);
			acResultsDiv.css("display","block");
			
			// for all divs in results
			var divs = $(acResultsId + " > div");
		
			// on mouse over clean previous selected and set a new one
			divs.mouseover( function() {
				divs.each(function(){ this.className = "unselected"; });
				this.className = "selected";
			})
		
			// on click copy the result text to the search field and hide
			divs.click( function() {
				acSearchField.val(this.childNodes[0].nodeValue);
				clearAutoComplete();
			});

		} else {
			clearAutoComplete();
		}
	});
}

// clear auto complete box
function clearAutoComplete()
{
	acResultsDiv.html('');
	acResultsDiv.css("display","none");
}

// reposition the results div accordingly to the search field
function repositionResultsDiv()
{
	// get the field position
	var sf_pos    = acSearchField.offset();
	var sf_top    = sf_pos.top;
	var sf_left   = sf_pos.left;

	// get the field size
	var sf_height = acSearchField.height();
	var sf_width  = acSearchField.width();

	// apply the css styles - optimized for Firefox
	acResultsDiv.css("position","absolute");
	acResultsDiv.css("left", sf_left - 2);
	acResultsDiv.css("top", sf_top + sf_height + 5);
	acResultsDiv.css("width", sf_width - 2);
}


// treat up and down key strokes defining the next selected element
function updownArrow(keyCode) {
	if(keyCode == 40 || keyCode == 38){

		if(keyCode == 38){ // keyUp
			if(acListCurrent == 0 || acListCurrent == -1){
				acListCurrent = acListTotal-1;
			}else{
				acListCurrent--;
			}
		} else { // keyDown
			if(acListCurrent == acListTotal-1){
				acListCurrent = 0;
			}else {
				acListCurrent++;
			}
		}

		// loop through each result div applying the correct style
		acResultsDiv.children().each(function(i){
			if(i == acListCurrent){
				acSearchField.val(this.childNodes[0].nodeValue);
				this.className = "selected";
			} else {
				this.className = "unselected";
			}
		});

		return true;
	} else {
		// reset
		acListCurrent = -1;
		return false;
	}
}