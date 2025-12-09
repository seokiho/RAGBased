@echo off
echo GitHub에 업로드 중...
git add .
git commit -m "Update files: %date% %time%"
git push origin main
echo 업로드 완료!
