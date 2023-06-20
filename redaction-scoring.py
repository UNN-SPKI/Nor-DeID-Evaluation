from string2string.alignment import NeedlemanWunsch
nw = NeedlemanWunsch()
source = "Pasienten , Olaug Nordmann , ble raskt dårlig igjen etter kontroll 2. mars".split()
target = "Pasienten , <NAME> , ble raskt dårlig igjen etter kontroll <DATE>".split()
aligned_source, aligned_target = nw.get_alignment(source, target)
nw.print_alignment(aligned_source, aligned_target)