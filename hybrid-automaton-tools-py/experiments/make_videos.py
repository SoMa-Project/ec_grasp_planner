#!/usr/bin/env python
import glob
import subprocess
import tempfile
import shutil
import os
import yaml
import argparse
import rosbag

# use mkvmerge GUI to concat videos (without re-encoding)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
      description=
        'Creates a video from sensor_msgs/Image messages from a bagfile. '
        'This script uses the extract_images binary to extract color images '
        'from bagfiles and calls ffmpeg afterwards to combine them together '
        'to form a video. Note that ffmpeg must be installed on your system.')
    parser.add_argument('directory', type=str, default=".", help='directory that contains bag files')
    parser.add_argument('--fps', type=float, help='frames per second, if nothing is specified this will be inferred from the bag file')
    args = parser.parse_args()
    
    generated_files = []
    
    # go through all bags
    for f in glob.glob(args.directory + "/*_video.bag"):
        filename_without_ext = os.path.splitext(f)[-2]
        tmp_output_filename = filename_without_ext + "TMP.mp4"
        final_output_filename = filename_without_ext + ".mp4"
        
        # the two video file names that will be merged side-by-side
        temp_name1 = next(tempfile._get_candidate_names()) + ".mp4"
        temp_name2 = next(tempfile._get_candidate_names()) + ".mp4"
        
        bag = rosbag.Bag(f, 'r')
        baginfo = bag.get_type_and_topic_info()[1]
        topics = baginfo.keys()
        print topics
        # ['/camera1/image_color', '/camera2/image_color']
        
        tmp_names = zip(topics, [temp_name1, temp_name2])
        for topic, videoname in tmp_names:
            tmp_dir = tempfile.mkdtemp()
            
            cmd = ["rosrun", "bag_tools", "extract_images" , tmp_dir, "jpg", topic, f]
            subprocess.call(cmd)

            # rename images
            images = glob.glob(tmp_dir + '/*.jpg')
            images.sort()
            for i, image in enumerate(images):
                shutil.move(image, tmp_dir + '/img-' + str(i) + '.jpg')

            fps = baginfo[topic].frequency if args.fps is None else args.fps
            cmd = ["ffmpeg", "-f", "image2", "-r", str(fps), "-i", tmp_dir + "/img-%d.jpg", videoname]
            subprocess.call(cmd)
            
            shutil.rmtree(tmp_dir)
        
        # merge both videos side-by-side
        if (len(topics) > 1):
            cmd = ["ffmpeg", "-i",  temp_name2, "-i", temp_name1, "-filter_complex", "[0:v]pad=iw*2:ih[int];[int][1:v]overlay=W/2:0[vid]", "-map", "[vid]", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", tmp_output_filename]
            subprocess.call(cmd)
        else:
            cmd = ["ffmpeg", "-i",  tmp_names[0][1], "-filter_complex", "[0:v]pad=iw*2:ih[vid]", "-map", "[vid]", "-c:v", "libx264", "-crf", "23", "-preset", "veryfast", tmp_output_filename]
            subprocess.call(cmd)
            #shutil.move(tmp_names[0][1], tmp_output_filename)
        
        # overlay text with param information
        params = yaml.load(open(filename_without_ext[:-6] + ".yml"))
        desired_params = {
            'edge_grasp': ['angle_of_sliding', 'edge_distance_factor', 'finger_inflation', 'sliding_speed', 'downward_force'],
            'surface_grasp': ['angle_of_sliding', 'finger_inflation', 'downward_force'],
        }
        filtered_params = { k: v for k, v in params.items() if k in desired_params[params['grasp_type']] }
        text_lines = ["%s = %f" % (k, v) for k, v in filtered_params.iteritems()]
        cmd = ["ffmpeg", "-i", tmp_output_filename, "-vf", "[in]drawbox=x=iw/2-(w/2):y=0:w=50:h=ih:color=black:t=max," + ','.join(["drawtext=fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf:text="+t+":fontcolor=white:fontsize=30:box=1:boxcolor=black@0.5:boxborderw=5:x=10:y=(h-"+str(i+1)+"*(text_h+20))" for i,t in enumerate(text_lines)]), "-codec:a", "copy", final_output_filename]
        subprocess.call(cmd)
        
        if len(topics) > 1:
            for n in tmp_names:
                os.remove(n[1])
        os.remove(tmp_output_filename)
        
        generated_files.append(final_output_filename)
    
    if len(generated_files) > 0:
        print "FINISHED. You can now merge the resulting files with this command:"
        generated_files = sorted(generated_files)
        print "mkvmerge -o full.mkv " + (' + '.join(generated_files[1:]))
