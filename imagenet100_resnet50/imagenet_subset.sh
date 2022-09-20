#!/bin/bash
input="./classes100.txt"
mkdir ./selected_100
while IFS= read -r line
do
   mkdir ./selected_100/$line
   tar xf ./train/$line.tar -C ./selected_100/$line
   echo "$line"
done < "$input"
