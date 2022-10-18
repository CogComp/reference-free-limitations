DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Download the WMT'19 sources, references, and submissions
wget http://ufallab.ms.mff.cuni.cz/~bojar/wmt19/wmt19-submitted-data-v3-txt-minimal.tgz
tar xzf wmt19-submitted-data-v3-txt-minimal.tgz
rm wmt19-submitted-data-v3-txt-minimal.tgz
mv wmt19-submitted-data-v3 ${DIR}

# Download the WMT'19 judgments
wget http://ufallab.ms.mff.cuni.cz/~bojar/wmt19-metrics-task-package.tgz
tar xzf wmt19-metrics-task-package.tgz
rm wmt19-metrics-task-package.tgz
mv wmt19-metrics-task-package ${DIR}
