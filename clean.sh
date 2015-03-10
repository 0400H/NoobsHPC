#! /bin/bash

dos2unix `find .`

files=`find . -name \*.h* ! -path "./build/*" -print -o -name \*.cpp ! -path "./build/*" -print`

for file in ${files}
do
	echo ${file}
	sed -i 's/[ \t]*$//' ${file}
done

