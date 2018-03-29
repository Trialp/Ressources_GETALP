#!/bin/bash

function dumpCommands {
for status in modified "not finished" deleted finished "classified finished"; do
	for lg in deu fra ita slv axi; do
		MYCOMMAND="select f2.value, f1.value, f.value, f.entryid from idxlexalp${lg} as f1 " 
   	MYCOMMAND=${MYCOMMAND}" left join idxlexalp${lg} as f2 on f1.entryid=f2.entryid " 
   	MYCOMMAND=${MYCOMMAND}" inner join idxlexalp${lg} as f on f2.entryid=f.entryid "
   	MYCOMMAND=${MYCOMMAND}" where f1.key='cdm-headword' and f2.key='lexalp-domain' and f2.value='4.1' and f.key='cdm-contribution-status' and f.value='${status}';"
  	echo "select text('${lg} ${status}');"
  	echo $MYCOMMAND 
	done
done
}


dumpCommands