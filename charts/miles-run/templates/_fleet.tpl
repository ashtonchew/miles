{{- define "miles-run.colocateRole" -}}
{{- $colocate := .context.Values.run.colocate | default dict -}}
{{- if $colocate.enabled -}}
{{- if eq .name $colocate.engineFleet }}engine{{ else if eq .name $colocate.trainerFleet }}trainer{{ end }}
{{- end }}
{{- end }}

{{- define "miles-run.fleet" -}}
{{- $context := .context }}
{{- $fleet := .fleet }}
{{- $role := include "miles-run.colocateRole" (dict "context" $context "name" $fleet.name) }}
{{- $gated := eq $role "engine" }}
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
        {{- include "miles-run.podDefaultsFor" (dict "context" $context "gated" $gated) | nindent 8 }}
        {{- if $role }}
        hostIPC: true
        {{- end }}
        {{- if $gated }}
        schedulingGates:
          - name: "miles.radixark.io/colocate-pairing"
        {{- end }}
        containers:
          - name: {{ $fleet.containerName | default "worker" | quote }}
            {{- include "miles-run.containerDefaults" $context | nindent 12 }}
            command:
              {{- range $fleet.command }}
              - {{ . | quote }}
              {{- end }}
            {{- $entry := $fleet }}
            {{- if $gated }}
            {{- $entry = merge (dict "env" (merge (dict "NVIDIA_VISIBLE_DEVICES" "all") (deepCopy ($fleet.env | default dict)))) (deepCopy $fleet) }}
            {{- end }}
            {{- with include "miles-run.env" (dict "context" $context "entry" $entry) }}
            {{- . | nindent 12 }}
            {{- end }}
            {{- with $fleet.ports }}
            ports:
              {{- range . }}
              - name: {{ .name | quote }}
                containerPort: {{ .port }}
              {{- end }}
            {{- end }}
            {{- $resources := default dict $fleet.resources }}
            {{- if $gated }}
            {{- $limits := deepCopy ($resources.limits | default dict) }}
            {{- $resources = deepCopy $resources }}
            {{- $_ := set $limits "nvidia.com/gpu" 0 }}
            {{- $ignored := set $resources "limits" $limits }}
            {{- end }}
            resources:
              {{- toYaml $resources | nindent 14 }}
        {{- $volumes := compact (list (include "miles-common.sharedStorageVolume" $context | trim) (include "miles-run.nodeLocalVolume" $context | trim)) | join "\n" }}
        {{- with $volumes }}
        volumes:
          {{- . | nindent 10 }}
        {{- end }}
{{- end }}
