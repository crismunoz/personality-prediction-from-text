echo off
set data_name=dataset_regular_answers
set nsents=3
set total_chunk=8
for /L %%c in (0,1,%total_chunk%) do (
    echo ...Starting console %%c
    START "console %%c" cmd /c "C:\Users\Cristian\anaconda3\condabin\conda.bat activate personality && python inference.py --chunk_id %%c --total_chunk %total_chunk% --dataset_path data_input/%data_name%.xlsx --output_path data_output_%data_name%_%nsents% --max_sentences %nsents%"
    )
pause