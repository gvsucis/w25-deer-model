#fast api
Start-Process powershell -ArgumentList "cd backend; venv\Scripts\Activate; uvicorn main:app --reload"

#nextjs app
Start-Process powershell -ArgumentList "cd frontend/buck3d; npm run dev"