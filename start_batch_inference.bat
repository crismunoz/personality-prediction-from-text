echo off
set dataset_path=C:\Users\Cristian\Documents\HolisticAI\repos\neural_nets_personality\outputs\organized_text\trait_activating_questions_clean.csv
set total_workers=10
set total_chunks=20
for /L %%c in (0,1,%total_workers%) do (
    echo ...Starting console %%c
    START "console %%c" cmd /c "C:\Users\Cristian\anaconda3\condabin\conda.bat activate personality && python batch_inference.py --chunk_id %%c --total_chunks %total_chunks% --dataset_path %dataset_path%"
    )
pause