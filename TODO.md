- [x] ściągnij datasety oxford pets oraz stanford
- [x] przygotuj transformacje zdjęć dla każdego z datasetów
- [x] odpal kod pod trening unimodalny np. na cifar10 - sprawdź czy otrzymam ~ 93% test acc
- [x] rozbuduj trainera o mapowanie ze słowników
- [] rozbuduj bardziej trainera o mapowanie ze słowników
- [] dodaj opcje schedulera
- [] zainstaluj conde jeszcze raz i biblioteki do niej
- [x] dodaj transformacja do każdego z datasetów
- [] dodaj opcję logowania do wandb
- [] postaw prywatnego wandb


- [] sprawdź jak klasyfikacja zależy od mieszania prawej częsci obrazu między dwoma datasetami (OXFORD PETS i CVSD)
- [x] sprawdź jak klasyfikacja zależy od zaszumiania prawej częsci obrazu między dwoma datasetami (CIFAR10 i CVSD)
- [] zrób tak żeby nie nadpisywać, ani nie korzystać z tych samych indeksów
- [] czy model ma lepsze acc na zbiorze treningowym na zdjęciach z prawą częścią niezaszumioną czy nie? czy grupę mniejszościową stanowią zdjęcia z zaszumioną prawą częścią?
- [x] eksperyment z blurem gdzie dwa datasety to CIFAR10 + zamiast blurra dawaj biały szum
- [] ogarnij rozszerzenia - usuń niepotrzebne
- [] zapisywanie logów, checkpointów, wykresów i rysunków w jednym podfolderze danego runu
- [] loguj accuracy i loss per class