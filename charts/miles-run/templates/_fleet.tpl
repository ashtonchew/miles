{{- define "miles-run.fleet" -}}
{{- $context := .context }}
{{- $fleet := .fleet }}
{{- $name := include "miles-run.componentName" (dict "context" $context "component" $fleet.name) }}
{{- $labels := dict "context" $context "component" $fleet.name }}
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: {{ $name | quote }}
  namespace: {{ $context.Release.Namespace | quote }}
  labels:
    {{- include "miles-run.labels" $labels | nindent 4 }}
spec:
  replicas: {{ default 1 $fleet.replicas }}
  startupPolicy: LeaderCreated
  leaderWorkerTemplate:
    size: {{ default 1 $fleet.size }}
    restartPolicy: RecreateGroupOnPodRestart
    workerTemplate:
      metadata:
        labels:
          {{- include "miles-run.labels" $labels | nindent 10 }}
          {{- with $fleet.specName }}
          miles.radixark.io/spec-name: {{ . | quote }}
          {{- end }}
        {{- with $fleet.meta }}
        annotations:
          {{- range $key, $value := . }}
          miles.radixark.io/meta-{{ $key }}: {{ $value | quote }}
          {{- end }}
        {{- end }}
      spec:
        {{- include "miles-run.podDefaults" $context | nindent 8 }}
        containers:
          - name: {{ $fleet.containerName | default "worker" | quote }}
            {{- include "miles-run.containerDefaults" $context | nindent 12 }}
            command:
              {{- range $fleet.command }}
              - {{ . | quote }}
              {{- end }}
            {{- with include "miles-run.env" (dict "context" $context "entry" $fleet) }}
            {{- . | nindent 12 }}
            {{- end }}
            {{- with $fleet.ports }}
            ports:
              {{- range . }}
              - name: {{ .name | quote }}
                containerPort: {{ .port }}
              {{- end }}
            {{- end }}
            resources:
              {{- toYaml (default dict $fleet.resources) | nindent 14 }}
        {{- $volumes := compact (list (include "miles-common.sharedStorageVolume" $context | trim) (include "miles-run.nodeLocalVolume" $context | trim)) | join "\n" }}
        {{- with $volumes }}
        volumes:
          {{- . | nindent 10 }}
        {{- end }}
{{- end }}
