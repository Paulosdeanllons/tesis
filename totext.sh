#!/bin/bash

#Calcula MD5 del fichero para no procesar duplicados
for file in input/*.pdf; do nuevo=($(md5sum "$file"));
#Comprueba si existe fichero de log para evitar errores. Si no existe, lo crea
if [ ! -f listado.log ]; then touch listado.log; fi
#Comprueba que la entrada no sea duplicada - Si lo es ¡no hace nada!
if grep -Fq "$nuevo" listado.log
then :
#Genera el archivo de texto y crea una entrada del log con fecha md5 y nombre de fichero
else
	pdftotext "$file";
    timestamp=$(date --iso-8601=seconds)
    echo "$timestamp" "$nuevo" "$file" >> listado.log;
fi
done
#Comprueba si hay salida y si la hay la mueve al directorio de procesado
salida=(`find ./input/ -name "*.txt"`)
if [ ${#salida[@]} -gt 0 ]; then 
    mv input/*.txt tmp
else :
fi

