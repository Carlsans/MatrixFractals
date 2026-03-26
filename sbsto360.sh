ffmpeg -i /home/carl/PycharmProjects/MatrixFractals/vr2.mp4 \
-filter_complex "\
[0:v]crop=iw/2:ih:0:0[left]; \
[0:v]crop=iw/2:ih:iw/2:0[right]; \
[left][right]vstack=inputs=2[stereo]" \
-map "[stereo]" -c:v libx264 -crf 18 -preset slow \
-pix_fmt yuv420p output_360_sbs_tb.mp4