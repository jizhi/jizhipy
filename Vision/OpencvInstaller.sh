#! /usr/bin/env bash
pip install --user opencv-python
pip install --user opencv-contrib-python
#----------------------------------------
version=`pip freeze | grep opencv-python`
version=${version#*==}
version=${version%.*}
if [ ! -f ${version}.zip ]; then
	wget https://github.com/opencv/opencv/archive/${version}.zip
fi
unzip ${version}.zip
if [ -d _source ]; then
	/bin/rm -rf _source
fi
mv opencv-${version} _source
#----------------------------------------
echo -e "import data\nimport samples" > _source/__init__.py
#----------------------------------------
files=`ls _source/data`
string=""
for f in ${files}; do
	if [ -d "_source/data/"${f} ]; then
		string=${string}"import "${f}"\n"
		touch _source/data/${f}/__init__.py
	fi
done
echo -e ${string} > _source/data/__init__.py
#----------------------------------------
echo "import dnn" > _source/samples/__init__.py
echo "import face_detector" > _source/samples/dnn/__init__.py
touch _source/samples/dnn/face_detector/__init__.py
#----------------------------------------
location=`pip show opencv-python | grep Location:`
location=${location#*Location:}"/cv2/"
if [ -d ${location}source ]; then
	/bin/rm -rf ${location}source
fi
mv _source ${location}source
content=`cat ${location}__init__.py | grep "import source"`
if !([[ "${content}" =~ "import source" ]]); then
	echo -e "\nimport source" >> ${location}__init__.py
	cat -v ${location}__init__.py | tr -d '^M'  > .OpencvInstaller_temp.py
	mv .OpencvInstaller_temp.py ${location}__init__.py
fi
#----------------------------------------
cp ${location}samples/dnn/tf_*.py ~/bin/
#----------------------------------------
echo "Installed  opencv-"${version}"  to"${location}" named  source"
/bin/rm ${version}.zip
