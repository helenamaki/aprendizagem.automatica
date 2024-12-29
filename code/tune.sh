# Create a log filename with the current date and time
LOG_DIR="../logs"
mkdir -p "$LOG_DIR"  # Ensure the logs directory exists
LOG_FILE="$LOG_DIR/$(date '+%Y%m%d_%H%M%S').log"

# Start logging: Save current stdout and stderr, then redirect to log file
exec 3>&1 4>&2       # Save current stdout (fd 3) and stderr (fd 4)
exec > >(tee -a "$LOG_FILE") 2>&1
python ordinalAdoptionPrediction.py
python ordinalAdoptionPrediction2.py
python ordinalSpecialized.py
python binarySpecialized.py
# Example logged commands
echo "This is a logged informational message."
ls non_existent_file  # Generates an error message

# Stop logging: Restore original stdout and stderr
exec 1>&3 2>&4        # Restore stdout (fd 1) and stderr (fd 2)
exec 3>&- 4>&-        # Close temporary file descriptors

python ~/scripts/processAlert.py "tune.sh" "$LOG_FILE"
