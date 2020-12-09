#!/bin/env shs

ls | grep pdf > listado

for file in *.pdf; do 
	pdftotext "$file"; 
done

