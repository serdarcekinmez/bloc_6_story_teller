docker run -it \
-v "$(pwd):/home/app" \
-p 80:8000 \
-e PORT=80 \
serdarcekinmez/story_webapp