#!/bin/bash
exec_path=$DATA2/deepcut-cnn/python/pose/est_pose_handtool_videos.py
root=$DATA2/100_handtool_videos
tools=('axe' 'barbell' 'carjack' 'hammer' 'hoe')
nvideos=10
extension='mp4'
for tool in ${tools[@]}
do
	for i in $(seq -f "%04g" 1 $nvideos)
	do
		video_name=${tool}_${i} 
		frames_dir=${root}/${tool}/frames
		out_dir=${root}/${tool}/deepercut
		info_path=${root}/${tool}/videos_info.pkl
		python ${exec_path} ${video_name} ${frames_dir} ${out_dir} ${info_path}
	done
done
