function filterVolumes() {

	var selected = document.getElementById('SourceLang');
	var lang = selected[selected.selectedIndex].value;
	
	var volumes = document.getElementById('VolumesDiv');
	var id = 'Volumes_' + lang;
	
	var i=0;
	while(i<volumes.childNodes.length) {
		child = volumes.childNodes[i];
		if (child.id!=id) {
			volumes.removeChild(child);
		}
		else {
			i++;
		}
	}
	return true;
}