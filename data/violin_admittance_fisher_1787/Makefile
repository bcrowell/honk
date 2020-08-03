default:
	perl -0777 -pe 's/"\s+cy="/,/igs' fisher.svg | grep cx | sed -e 's/cx="/\[/g' | sed -e 's/"/\],/g' >points.txt
	@echo "Cut and paste from points.txt into generate_fisher.py."

