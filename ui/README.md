# Sleepwalker UI

Schlichte Browseroberfläche für EDF-Aufnahmen: Server verbinden, Session öffnen,
Modelle ausführen, Signale und Vorhersagen sichten, korrigieren und exportieren.
Der gesamte neue Code liegt in `ui/`. `sleepwalker/server/` bleibt unverändert.

[Vorschau des Viewers](screenshots/viewer.png) · [Einstieg](screenshots/entry.png)
[Intervallauswahl und Rechtsklick](screenshots/correction-menu.png) ·
[Korrekturdialog](screenshots/correction-dialog.png) · [Korrigierte Spuren](screenshots/corrections.png)
· [Zuletzt geöffnete Sessions](screenshots/sessions.png)
· [Neue Vorhersagespuren](screenshots/predictions.png)
(Browseraufnahmen mit synthetischen Testdaten).

## Start

Aus dem Hauptrepository (dem Verzeichnis mit `setup.py`), mit Python 3.10 oder neuer:

```bash
python -m pip install -r ui/requirements.txt
python -m ui
```

Browser: <http://localhost:6006/ui/>. Ohne Modellkonfiguration funktionieren die
EDF-Signalansicht und das Öffnen vorhandener Workspaces.

Für Inferenz dieselben Modellpakete und dieselbe separate Python-Umgebung wie
beim bisherigen API-Start verwenden:

```bash
python -m ui \
  --models sleepwalker/server/models \
  --inference-python /absoluter/pfad/zur/inference-venv/bin/python \
  --device cpu \
  --batch-size 1
```

Dieser Befehl **ersetzt den API-Start für diese Instanz**: Er startet API und GUI
gemeinsam. Einen bereits laufenden API-Prozess auf Port 6006 unverändert lassen
und zum Ausprobieren einen anderen Port und eine eigene temporäre Ablage wählen:

```bash
python -m ui --port 6007 --storage /tmp/sleepwalker-ui-6007
```

Die separate Ablage verhindert Konflikte mit der exklusiven Speicherbelegung
des bestehenden Servers. Für einen Host im LAN oder Cluster:

```bash
python -m ui --host 0.0.0.0 --port 6006 --storage /tmp/sleepwalker-ui
```

Modelloptionen können wie oben ergänzt werden. `--device cuda` verwendet die
entsprechend eingerichtete GPU-Inferenzumgebung. Die mitgelieferte
`requirements-inference.txt` richtet eine CPU-Umgebung mit `torch==2.13.0+cpu`
ein; dafür muss `--device cpu` verwendet werden. Ein Start mit `--device cuda`
und dieser CPU-Umgebung führt beim Modelllauf zu einem Fehler. Die Torch-Version
der angegebenen separaten Inferenzumgebung ist maßgeblich, auch wenn im Rechner
eine GPU vorhanden ist. Die GUI plant keine Clusterjobs.
Eine intern erreichbare Instanz kann über einen bestehenden Tunnel oder
Port-Forward im Browser geöffnet werden. Am Ziel muss ebenfalls `python -m ui`
laufen; eine alleinstehende REST-API stellt den Python-Resampler nicht über HTTP
bereit. Serverwechsel öffnen deshalb die GUI der Zielinstanz. Externe Zugriffe
verwenden die vorhandene Infrastruktur für Zugriffsschutz und TLS.

## Container

Aus dem Repository-Verzeichnis:

```bash
docker build -f ui/Dockerfile -t sleepwalker-ui .
docker run --rm -p 6006:6006 sleepwalker-ui
```

Das Standardimage enthält die API-/UI-Abhängigkeiten. Für Inferenz kann ein
vorhandenes Image mit kompatibler separater Inferenzumgebung als `BASE_IMAGE`
verwendet werden. Modelle einhängen und beim Start die Modelloptionen ergänzen;
die Python-Pfade müssen dabei innerhalb des Containers gültig sein:

```bash
docker build --build-arg BASE_IMAGE=mein-inferenz-image -f ui/Dockerfile -t sleepwalker-ui .
docker run --rm -p 6006:6006 -v /pfad/zu/modellen:/models:ro sleepwalker-ui \
  --host 0.0.0.0 --port 6006 --storage /tmp/sleepwalker-ui \
  --models /models --inference-python /opt/inference/bin/python --device cpu
```

Die separate Modellumgebung wird durch diesen UI-Build nicht eingerichtet.
Der Docker-Build wurde in dieser Arbeitsumgebung mangels Docker nicht ausgeführt.

## Bedienung

- Beim Start wird die aktuelle Instanz geprüft. Weitere Serveradressen lassen
  sich im Serverformular eingeben; nur diese Profile werden im Browser dauerhaft
  gespeichert.
- EDF auswählen und entweder nur öffnen oder nach dem Upload inferieren.
  Bei mehreren Modellen für dieselbe Aufgabe lässt sich jeweils eines auswählen.
  Weitere, noch nicht inferierte Aufgaben können im Viewer unter **Modelle …**
  gestartet werden. Bereits vorhandene Tasks werden nicht überschrieben.
- Die Modellauswahl zeigt die benötigten Eingänge. Kann die API für ein Modell
  keine passende Kanalzuordnung finden, wird dieses Modell mit seinen konkreten
  fehlenden Zuordnungen angezeigt; die übrigen ausgewählten Modelle werden
  separat erneut zum Start eingereicht. Abgewiesene Anfragen erzeugen dabei noch
  keinen Inferenzjob. Die Prüfung und Aliasauflösung bleiben vollständig beim
  bestehenden Server. Unter **Kanalzuordnung** und **Modelle …** sind die
  vorhandenen EDF-Kanalnamen sichtbar. Abweichende Namen oder Ableitungen werden
  nicht automatisch umbenannt. Die Auswahl sichtbarer Plotkanäle verändert die
  Modelleingänge nicht.
- **Kanäle …** bestimmt Auswahl und Reihenfolge. Nur ausgewählte Kanäle werden
  in den vorhandenen Resampler geladen.
- Im Signalplot ziehen, um ein Intervall zu markieren. Das Mauswerkzeug
  **Verschieben** bewegt stattdessen die Zeitachse; **Markieren** ist der Standard.
  Mit dem Mausrad zoomen. Die unteren Spuren verwenden denselben Zeitbereich.
  Die Navigation verändert ausschließlich die Zeitachse; die Amplitudenachsen
  bleiben beim Ziehen und Zoomen gesperrt. Pfeiltasten links/rechts wechseln das
  sichtbare Fenster. Die Zeitachsen aller Kanäle sind direkt auf 0 bis zum Ende
  der Aufnahme begrenzt. Herauszoomen kann höchstens die gesamte EDF zeigen;
  Verschieben am Rand behält die Fensterbreite bei. Die Grenzen werden beim
  Wechsel der EDF oder der sichtbaren Kanäle neu gesetzt.
- Einen Klassenabschnitt anklicken, um ihn auszuwählen und einzurahmen. Ziehen
  markiert ein Intervall auch direkt in einer Vorhersagespur. Ein Rechtsklick
  auf die Auswahl öffnet das Menü zum Korrigieren oder Bearbeiten. Alternativ
  öffnet **Auswahl bearbeiten …** den Dialog; die fokussierte Timeline unterstützt
  außerdem **Umschalt+F10**. Ein Klick auf den Tasknamen klappt dessen einzelne
  Klassenspuren auf. Pfeiltasten oben/unten auf der fokussierten Timeline wählen
  die Aufgabe. Aufgaben können auch im Detailpanel ausgewählt werden.
- Vorhersagen richten sich nach `kind` und `resolution` der API. Die derzeitigen
  Modellpakete liefern epochenbasierte Klassen auch für Arousal, Atmung und
  Desaturation. Lücken sind fehlende Vorhersagen und keine negativen Klassen.
- Die Spuren verwenden dünne graue Grundlinien statt Hintergrundbalken. Aktive
  Klassen erscheinen als kräftige, flache Segmente: W gelb, N1 hellblau, N2 blau,
  N3 violett, REM magenta, Arousal orange, Apnoe rot und Desaturation türkis.
  Negative Klassen erhalten keine farbige Fläche. Klassennamen und eine kompakte
  Farblegende ergänzen die Farben. Kontinuierliche Aufgaben bleiben Kurven;
  ihre negativen Gegenklassen können über die aufgeklappten Klassenspuren
  betrachtet werden. Farben bleiben bei einem Wechsel der Aufgabenreihenfolge gleich.
- Im Dialog Aufgabe, Start, Ende und Klasse einstellen und **Speichern** wählen.
  Bei Epochen werden alle berührten vollständigen Epochen korrigiert; der Dialog
  zeigt ihre tatsächlichen Grenzen und Anzahl vor dem Speichern. Event- und
  kontinuierliche Tasks erhalten menschliche Ereignisse mit frei gewählten Grenzen.
  Korrigierte Klassen tragen ein kleines **M** in der Spur. Eine schwarze
  Umrandung kennzeichnet die aktuelle Auswahl; bei Intervallen ist sie gestrichelt.
  Die ursprüngliche
  Modellvorhersage und ihre Wahrscheinlichkeiten bleiben im Detailpanel sichtbar.
  Das frühere Korrekturformular im rechten Panel entfällt.
- Eigene Änderungen lassen sich direkt anklicken und über das Rechtsklickmenü
  bearbeiten oder löschen. Das Löschen einer Klassenkorrektur nimmt die ausgewählte
  Änderung zurück; eine frühere Korrektur derselben Epoche wird dadurch wieder
  sichtbar. Der Verlauf unterstützt weiterhin **Rückgängig**. Ereignisänderungen
  speichern zuerst den Ersatz und löschen anschließend das bisherige Ereignis.
  Schlägt dieser zweite Schritt fehl, wird der Ersatz zurückgenommen. Mehrere
  Epochen werden über einzelne vorhandene API-Aufrufe gespeichert; bei einem
  Teilfehler nennt der Dialog die bereits gespeicherte Anzahl und bleibt offen.
  Das Kontextmenü einer Intervallauswahl kann auch mehrere darin liegende
  Klassenkorrekturen gemeinsam zurücknehmen; eigene Ereignisse werden dabei nur
  gelöscht, wenn sie vollständig in der Auswahl liegen.
- Modellereignisse können über die vorhandene API nicht gelöscht oder in ihren
  Grenzen verändert werden. **Vorhersage löschen** ist deshalb im Menü deaktiviert
  und erklärt die Einschränkung. Korrekturen werden als menschliche Annotationen
  zusätzlich gespeichert und ändern die ursprünglichen Modellresultate nicht. Ein
  allgemeines Redo ist ebenfalls nicht Teil dieser API.
- **Exportieren** erstellt JSON-Ergebnisse mit wählbaren Bestandteilen oder ein
  vollständiges `.sleepwalker.zip` mit EDF, Ergebnissen, Metadaten und Verlauf.
  Ein solcher Workspace lässt sich auf der Startseite wieder öffnen. Downloads
  kommen direkt vom Export-Endpunkt, ohne große Dateien als Browser-Blob zu laden.
- **Schließen …** erlaubt Speichern, Behalten der temporären Session oder
  ausdrückliches Löschen. **Zur Startseite, Session behalten** verlässt die Ansicht,
  ohne die Aufzeichnung vom Server zu löschen.
- **F5** öffnet dieselbe Session automatisch. Ihre ID steht in der URL als
  `/ui/?session=…`; dieser Link kann auch in einem anderen Tab geöffnet werden.
  Zeitfenster, Kanalauswahl, Zeitpunkt, ausgewählte Aufgabe, aufgeklappte Spuren
  und Mauswerkzeug werden wiederhergestellt. Bekannte laufende Jobs werden wieder
  verbunden. Ein alter Verweis aus der bisherigen GUI wird beim ersten Laden übernommen.
- Die Startseite zeigt bis zu 20 **zuletzt geöffnete Sessions dieses Browsers**
  auf derselben Serveradresse. Der aktuelle Status wird über die vorhandenen
  Session-Endpunkte geprüft. **Fortsetzen** öffnet eine verfügbare Session;
  **Aus Liste entfernen** entfernt nur den Browsereintrag. Nach bewusstem Verlassen
  bleibt auch ein Reload auf der Startseite. Abgelaufene oder gelöschte Sessions
  werden entsprechend angezeigt. Eine vorübergehende Verbindungsstörung entfernt
  keine Verweise und erlaubt einen erneuten Versuch.
- Die browserweite Liste und deren Ansichtsmetadaten liegen in `localStorage`;
  die aktuelle Session und ihre Ansicht zusätzlich pro Tab in `sessionStorage`.
  Darin liegen Dateiname, Session-/Jobkennungen und UI-Auswahl, keine EDF-, Signal-
  oder Vorhersagearrays. Die Liste ist keine serverweite Auflistung: Die unveränderte
  API bietet dafür keinen Endpunkt. Sessions bleiben temporär und bestehen nur,
  solange der Server sie bereitstellt.

## Integration

`app.py` setzt eine äußere Starlette-Anwendung aus statischer Oberfläche, einem
Dash-Signalplot unter `/ui/plot/` und der unveränderten FastAPI-Anwendung zusammen.
Der ursprüngliche API-Lifespan läuft genau einmal; sämtliche ursprünglichen
Routen, HTTP-Verträge und die OpenAPI-Beschreibung bleiben erhalten.

Die Oberfläche verwendet kleine lokale ES-Module ohne Build-Schritt, externe
Schriftarten oder CDN-Abhängigkeiten. Der Signalplot ist ein eingebetteter
Dash-Graph. Eine auf dieselbe Origin beschränkte Nachrichtenverbindung übergibt
Zeitbereich, Auswahl und Plotstatus. `plots.py` ruft ausschließlich die vorhandene
Python-Schnittstelle `PlotlyService` für die Signale auf. Uploads sind direkte
Multipart-Übertragungen; alle fachlichen Operationen verwenden die bestehenden
API-Endpunkte. Jobstatus kommt über SSE, bei Streamproblemen über Polling.

Jeder Seitenaufruf erzeugt eine eigene `view_id`. Plotzustand wird beim Verlassen
explizit freigegeben und zusätzlich nach drei Minuten ohne GUI-Aktivität entfernt.
Ein sichtbarer Tab erneuert seinen Plotzugriff alle 45 Sekunden. Bei Wiederkehr
in einen länger verdeckten Tab wird eine freigegebene Ansicht neu erstellt.
Die bestehende Session-TTL und die API-eigene Löschung bleiben maßgeblich für die
Studie. Das Schließen eines Tabs garantiert keine sofortige Sessionlöschung.

Pro Instanz wird **ein Webprozess** verwendet. Die vorhandenen In-Memory-Stores
werden nicht zwischen Webworkern oder Replikaten geteilt. Die separate
Inferenzumgebung bleibt ein eigener Subprozess.

Der Resampler lädt vollständige ausgewählte Kanäle in den Server-RAM; die
Übertragung in den Browser wird auf ungefähr 2.000 Punkte pro Kanal begrenzt.
Zoom verwendet Resampler-Patches und lädt die EDF nicht erneut. Die
Resampler-Aufrufe enthalten den Zeitbereich aller Subplots, damit auch untere
Kanäle beim tiefen Zoom die Originalsamples erhalten. Die Plothöhe ist direkt an
die iframe-Höhe gebunden und kollabiert bei responsiven Dash-Updates nicht.
Schnelle Navigationsfolgen werden über 80 ms gebündelt; veraltete Statusantworten
setzen den sichtbaren Zeitbereich nicht zurück. Der Zeitcursor ist eine separate
DOM-Markierung und startet keinen zusätzlichen Plot-Neuaufbau.
Die bestehenden Prediction-Endpunkte werden nach dem sichtbaren Zeitbereich abgefragt. Dichtere
kontinuierliche Vorhersagen werden erst bei einem Fenster von höchstens 15 Minuten
geladen; die Gesamtansicht zeigt dazu einen Zoomhinweis. Die Signalansicht ist
zunächst für Single-EDF-Sessions vorgesehen, entsprechend dem vorhandenen Adapter.

## Tests

```bash
python -m pip install pytest httpx
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest ui/tests/test_integration.py -q
```

Die Behandlung abgewiesener Modelle lässt sich zusätzlich mit Node.js prüfen:

```bash
node ui/tests/test_inference.mjs
node ui/tests/test_corrections.mjs
node ui/tests/test_sessions.mjs
node ui/tests/test_colors.mjs
```

Die Integrationstests prüfen den unveränderten API-Vertrag, echte EDF-Uploads,
Resampling beim Zoom, voneinander getrennte Tabs, Kanalwechsel, verworfene alte
Kommandos und die Freigabe inaktiver Views. Für einen echten Browserdurchlauf:

```bash
python -m pip install playwright
python -m playwright install chromium
python -m ui.tests.browser_app --port 6019
# In einem zweiten Terminal:
python -m ui.tests.browser_check --url http://localhost:6019
python -m ui.tests.browser_plot_check --url http://localhost:6019
python -m ui.tests.browser_bounds_check --url http://localhost:6019
python -m ui.tests.browser_corrections_check --url http://localhost:6019
python -m ui.tests.browser_sessions_check --url http://localhost:6019
```

Die Browser-Testinstanz verwendet ausschließlich die EDF-Testdatei und explizite,
deterministische Testmodelle. Diese dienen zur Prüfung der GUI-Abläufe und sind
keine medizinischen Modelle. Der normale UI-Start enthält keine Testmodelle.

Verwendete Integrationsschnittstellen: [Dash clientside callbacks](https://dash.plotly.com/clientside-callbacks)
und die vorhandene [Plotly-Integrationsbeschreibung](../sleepwalker/server/PLOTLY.md).

Geprüfter Stand vom 16. Oktober 2026: 114 GUI-/API-Integrationstests bestanden,
vollständiger Chromium-Durchlauf einschließlich Kanalwechsel, Zoom, Korrekturen,
Verlauf, Workspace-Export/-Import, Reload und Löschung bestanden. Zusätzlich lief
das vorhandene Schlafstadienpaket über die GUI mit einer synthetischen EDF in der
bestehenden separaten Inferenzumgebung. Die vor Beginn vorhandenen 76 Dateien im
Serververzeichnis sind bytegleich erhalten geblieben. Ein entfernter Cluster wurde
in dieser Umgebung nicht praktisch getestet.

Ergänzend geprüft: fünf JavaScript-Tests für modellbezogene Kanalfehler,
Weiterverarbeitung der übrigen Modelle und veraltete Antworten. Der Browserlauf
enthält ein absichtlich inkompatibles Testmodell und prüft, dass die anderen
ausgewählten Modelle trotzdem Ergebnisse liefern.

Der zusätzliche Signal-Regressionslauf verwendet die synthetische Demo-EDF mit
acht Kanälen und prüft Originalsamples beim tiefen Zoom in jedem Subplot, echte
Mausbewegungen und Mausrad-Zoom, gesperrte Amplitudenachsen, verspätete Antworten,
Cursorbewegungen sowie schmale und breite Fenster. Die Plothöhe wird während
des gesamten Laufs in jedem Animationsframe kontrolliert.

Geprüft am 26. Oktober 2026: 45 GUI-/Resampler-Integrationstests, fünf
JavaScript-Tests, der vollständige GUI-Browserlauf und der zusätzliche
Signal-Regressionslauf bestanden. Im Signaltest blieb die Plothöhe durchgehend
bei 800 Pixeln; die längste gemessene Resampler-Antwort betrug lokal 114 ms.
Die 76 vor Beginn vorhandenen Serverdateien blieben bytegleich erhalten.

Die Korrekturfunktionen haben neun zusätzliche JavaScript-Tests und einen eigenen
Browserlauf mit echten Mausgesten: Signal- und Timeline-Intervalle, Klassenumrandung,
Epochenraster, Kontextmenüs, Abbruch, mehrere Epochen, Bearbeiten und Löschen,
kontinuierliche Tasks sowie ein absichtlich fehlgeschlagener Ereignisersatz.
Die Modellvorhersagen bleiben dabei nachweislich unverändert.
Zusätzlich bestanden 38 GUI-/Annotations-Integrationstests, die 14 JavaScript-Tests
und alle drei Browserläufe (Arbeitsablauf, Signalnavigation, Korrekturen). Für diese
Änderung wurden ausschließlich Dateien unter `ui/` bearbeitet.
Auch die laufende GUI auf Port 6009 wurde mit einer eigenen Demo-Session und allen
vier vorhandenen CPU-Modellen geprüft: Inferenz, Klassenauswahl per Maus,
Kontextkorrektur und anschließendes gemeinsames Löschen der Korrekturen bestanden.

Session-Wiederaufnahme: vier zusätzliche JavaScript-Tests, 16 GUI-Integrationstests
und der vollständige Arbeitsablauf mit automatischer Wiederaufnahme bestanden.
Der eigene Session-Browserlauf prüft F5 mit Zeitfenster und Kanalwahl, mehrere
Aufzeichnungen und Tabs, direkte Links, bewusstes Verlassen, lokales Entfernen,
abgelaufene Sessions und vorübergehende Verbindungsfehler. Dieser Lauf bestand
auch gegen die laufende GUI auf Port 6007 mit eigenen Test-Sessions.

Die neue Spurdarstellung wurde mit drei Tests für lesbare Textkontraste,
negative Klassen und feste Farben sowie pixelgenauen Browserprüfungen für
Segmente und freie Flächen an der Grundlinie getestet. 16 GUI-Integrationstests,
der vollständige GUI-Ablauf und der Korrektur-Browserlauf bestanden. Zusätzlich
wurde die laufende GUI auf Port 6007 mit allen vier vorhandenen CPU-Modellen
und einer eigenen Demo-Session einschließlich Korrektur und Löschung geprüft.

Die Aufnahmegrenzen wurden mit echtem Ziehen und Mausrad-Zoom an beiden Enden,
auf allen acht Kanälen, in der Gesamtansicht und nach Kanal-/Sessionwechseln
geprüft. Der Lauf mit EDF-Längen von 690 und 3600 Sekunden bestand auch auf der
laufenden GUI auf Port 6007. Zusätzlich bestanden die 16 GUI-Integrationstests
und der bestehende Navigationstest mit verspäteten Resampler-Antworten.
