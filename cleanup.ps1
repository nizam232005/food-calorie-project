$filesToRemove = @(
    "add_mango.py", "add_spaghetti.py", "calibrate_labels.py", "check_db.py", 
    "check_gemini.py", "check_schema_definitive.py", "debug_model.py", 
    "debug_usda_api.py", "diagnostic.py", "final_check.py", "fix_folders.py", 
    "food_check.py", "migrate_db.py", "prep_tm_data.py", "probe_model.py", 
    "verify_final.py", "verify_migration.py", "test.py", "test_clarifai_api.py", 
    "test_nutrition.py", "test_predictor.py", "test_tm_model.py", "test_yolo.py", 
    "extract_food101.py", "train_cnn.py", "debug_results.txt", "test_food.jpg.png", 
    "write_test.txt", "keys"
)

foreach ($file in $filesToRemove) {
    if (Test-Path $file) {
        Remove-Item $file -Force -ErrorAction SilentlyContinue
        Write-Host "Removed: $file"
    } else {
        Write-Host "Not found: $file"
    }
}
