@__hip_cuid_eabfcf1a4bdc33de = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [30 x ptr] [ptr @syncthreads(), ptr @load_acquire_agent(unsigned long AS1*), ptr @load_relaxed_agent(unsigned long AS1*), ptr @load_acquire_system(unsigned long AS1*), ptr @load_relaxed_system(unsigned long AS1*), ptr @store_relaxed_agent(unsigned long AS1*), ptr @store_release_agent(unsigned long AS1*), ptr @store_relaxed_system(unsigned long AS1*), ptr @store_release_system(unsigned long AS1*), ptr @atom_add_acqrel_agent(int AS1*, int AS1*), ptr @red_add_release_agent(int AS1*, int AS1*), ptr @atom_add_acqrel_system(int AS1*, int AS1*), ptr @atom_add_acquire_agent(int AS1*, int AS1*), ptr @atom_add_relaxed_agent(int AS1*, int AS1*), ptr @load_acquire_workgroup(unsigned long AS1*), ptr @load_relaxed_workgroup(unsigned long AS1*), ptr @red_add_release_system(int AS1*, int AS1*), ptr @atom_add_acquire_system(int AS1*, int AS1*), ptr @atom_add_relaxed_system(int AS1*, int AS1*), ptr @store_relaxed_workgroup(unsigned long AS1*), ptr @store_release_workgroup(unsigned long AS1*), ptr @atom_cas_acqrel_relaxed_agent(int AS1*, int AS1*, int AS1*), ptr @atom_cas_acqrel_relaxed_system(int AS1*, int AS1*, int AS1*), ptr @atom_cas_acquire_relaxed_agent(int AS1*, int AS1*, int AS1*), ptr @atom_cas_relaxed_relaxed_agent(int AS1*, int AS1*, int AS1*), ptr @atom_cas_release_relaxed_agent(int AS1*, int AS1*, int AS1*), ptr @atom_cas_acquire_relaxed_system(int AS1*, int AS1*, int AS1*), ptr @atom_cas_relaxed_relaxed_system(int AS1*, int AS1*, int AS1*), ptr @atom_cas_release_relaxed_system(int AS1*, int AS1*, int AS1*), ptr addrspacecast (ptr addrspace(1) @__hip_cuid_eabfcf1a4bdc33de to ptr)], section "llvm.metadata"

define internal noundef i64 @load_acquire_workgroup(unsigned long AS1*)(ptr addrspace(1) noundef readonly captures(none) %input) #0 !dbg !12 {
entry:
  %0 = load atomic i64, ptr addrspace(1) %input syncscope("workgroup") acquire, align 8, !dbg !16
  ret i64 %0, !dbg !17
}

define internal noundef i64 @load_relaxed_workgroup(unsigned long AS1*)(ptr addrspace(1) noundef readonly captures(none) %input) #0 !dbg !18 {
entry:
  %0 = load atomic i64, ptr addrspace(1) %input syncscope("workgroup") monotonic, align 8, !dbg !19
  ret i64 %0, !dbg !20
}

define internal noundef i64 @load_acquire_agent(unsigned long AS1*)(ptr addrspace(1) noundef readonly captures(none) %input) #0 !dbg !21 {
entry:
  %0 = load atomic i64, ptr addrspace(1) %input syncscope("agent") acquire, align 8, !dbg !22
  ret i64 %0, !dbg !23
}

define internal noundef i64 @load_relaxed_agent(unsigned long AS1*)(ptr addrspace(1) noundef readonly captures(none) %input) #0 !dbg !24 {
entry:
  %0 = load atomic i64, ptr addrspace(1) %input syncscope("agent") monotonic, align 8, !dbg !25
  ret i64 %0, !dbg !26
}

define internal noundef i64 @load_acquire_system(unsigned long AS1*)(ptr addrspace(1) noundef readonly captures(none) %input) #0 !dbg !27 {
entry:
  %0 = load atomic i64, ptr addrspace(1) %input acquire, align 8, !dbg !28
  ret i64 %0, !dbg !29
}

define internal noundef i64 @load_relaxed_system(unsigned long AS1*)(ptr addrspace(1) noundef readonly captures(none) %input) #0 !dbg !30 {
entry:
  %0 = load atomic i64, ptr addrspace(1) %input monotonic, align 8, !dbg !31
  ret i64 %0, !dbg !32
}

define internal noundef i64 @store_release_workgroup(unsigned long AS1*)(ptr addrspace(1) noundef writeonly captures(none) %input) #0 !dbg !33 {
entry:
  store atomic i64 1, ptr addrspace(1) %input syncscope("workgroup") release, align 8, !dbg !34
  ret i64 1, !dbg !35
}

define internal noundef i64 @store_relaxed_workgroup(unsigned long AS1*)(ptr addrspace(1) noundef writeonly captures(none) %input) #0 !dbg !36 {
entry:
  store atomic i64 1, ptr addrspace(1) %input syncscope("workgroup") monotonic, align 8, !dbg !37
  ret i64 1, !dbg !38
}

define internal noundef i64 @store_release_agent(unsigned long AS1*)(ptr addrspace(1) noundef writeonly captures(none) %input) #0 !dbg !39 {
entry:
  store atomic i64 1, ptr addrspace(1) %input syncscope("agent") release, align 8, !dbg !40
  ret i64 1, !dbg !41
}

define internal noundef i64 @store_relaxed_agent(unsigned long AS1*)(ptr addrspace(1) noundef writeonly captures(none) %input) #0 !dbg !42 {
entry:
  store atomic i64 1, ptr addrspace(1) %input syncscope("agent") monotonic, align 8, !dbg !43
  ret i64 1, !dbg !44
}

define internal noundef i64 @store_release_system(unsigned long AS1*)(ptr addrspace(1) noundef writeonly captures(none) %input) #0 !dbg !45 {
entry:
  store atomic i64 1, ptr addrspace(1) %input release, align 8, !dbg !46
  ret i64 1, !dbg !47
}

define internal noundef i64 @store_relaxed_system(unsigned long AS1*)(ptr addrspace(1) noundef writeonly captures(none) %input) #0 !dbg !48 {
entry:
  store atomic i64 1, ptr addrspace(1) %input monotonic, align 8, !dbg !49
  ret i64 1, !dbg !50
}

define internal noundef i64 @syncthreads()() #1 !dbg !51 {
entry:
  fence syncscope("workgroup") release, !dbg !52
  tail call void @llvm.amdgcn.s.barrier(), !dbg !60
  fence syncscope("workgroup") acquire, !dbg !61
  ret i64 0, !dbg !62
}

define internal noundef i32 @red_add_release_agent(int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !63 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !64
  %1 = atomicrmw add ptr addrspace(1) %atomic_address, i32 %0 syncscope("agent") release, align 4, !dbg !69
  ret i32 %1, !dbg !70
}

define internal noundef i32 @red_add_release_system(int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !71 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !72
  %1 = atomicrmw add ptr addrspace(1) %atomic_address, i32 %0 release, align 4, !dbg !73
  ret i32 %1, !dbg !74
}

define internal noundef i32 @atom_add_acquire_agent(int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !75 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !76
  %1 = atomicrmw add ptr addrspace(1) %atomic_address, i32 %0 syncscope("agent") acquire, align 4, !dbg !77
  ret i32 %1, !dbg !78
}

define internal noundef i32 @atom_add_relaxed_agent(int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !79 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !80
  %1 = atomicrmw add ptr addrspace(1) %atomic_address, i32 %0 syncscope("agent") monotonic, align 4, !dbg !81
  ret i32 %1, !dbg !82
}

define internal noundef i32 @atom_add_acqrel_agent(int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !83 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !84
  %1 = atomicrmw add ptr addrspace(1) %atomic_address, i32 %0 syncscope("agent") acq_rel, align 4, !dbg !85
  ret i32 %1, !dbg !86
}

define internal noundef i32 @atom_add_acquire_system(int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !87 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !88
  %1 = atomicrmw add ptr addrspace(1) %atomic_address, i32 %0 acquire, align 4, !dbg !89
  ret i32 %1, !dbg !90
}

define internal noundef i32 @atom_add_relaxed_system(int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !91 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !92
  %1 = atomicrmw add ptr addrspace(1) %atomic_address, i32 %0 monotonic, align 4, !dbg !93
  ret i32 %1, !dbg !94
}

define internal noundef i32 @atom_add_acqrel_system(int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !95 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !96
  %1 = atomicrmw add ptr addrspace(1) %atomic_address, i32 %0 acq_rel, align 4, !dbg !97
  ret i32 %1, !dbg !98
}

define internal noundef range(i64 0, 2) i64 @atom_cas_acquire_relaxed_agent(int AS1*, int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef captures(none) %compare, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !99 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !100
  %1 = load i32, ptr addrspace(1) %compare, align 4, !dbg !101
  %2 = cmpxchg ptr addrspace(1) %atomic_address, i32 %1, i32 %0 syncscope("agent") acquire monotonic, align 4, !dbg !101
  %3 = extractvalue { i32, i1 } %2, 1, !dbg !101
  br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !101

cmpxchg.store_expected:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, ptr addrspace(1) %compare, align 4, !dbg !101
  br label %cmpxchg.continue, !dbg !101

cmpxchg.continue:
  %conv = zext i1 %3 to i64, !dbg !101
  ret i64 %conv, !dbg !102
}

define internal noundef range(i64 0, 2) i64 @atom_cas_release_relaxed_agent(int AS1*, int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef captures(none) %compare, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !103 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !104
  %1 = load i32, ptr addrspace(1) %compare, align 4, !dbg !105
  %2 = cmpxchg ptr addrspace(1) %atomic_address, i32 %1, i32 %0 syncscope("agent") release monotonic, align 4, !dbg !105
  %3 = extractvalue { i32, i1 } %2, 1, !dbg !105
  br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !105

cmpxchg.store_expected:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, ptr addrspace(1) %compare, align 4, !dbg !105
  br label %cmpxchg.continue, !dbg !105

cmpxchg.continue:
  %conv = zext i1 %3 to i64, !dbg !105
  ret i64 %conv, !dbg !106
}

define internal noundef range(i64 0, 2) i64 @atom_cas_relaxed_relaxed_agent(int AS1*, int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef captures(none) %compare, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !107 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !108
  %1 = load i32, ptr addrspace(1) %compare, align 4, !dbg !109
  %2 = cmpxchg ptr addrspace(1) %atomic_address, i32 %1, i32 %0 syncscope("agent") monotonic monotonic, align 4, !dbg !109
  %3 = extractvalue { i32, i1 } %2, 1, !dbg !109
  br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !109

cmpxchg.store_expected:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, ptr addrspace(1) %compare, align 4, !dbg !109
  br label %cmpxchg.continue, !dbg !109

cmpxchg.continue:
  %conv = zext i1 %3 to i64, !dbg !109
  ret i64 %conv, !dbg !110
}

define internal noundef range(i64 0, 2) i64 @atom_cas_acqrel_relaxed_agent(int AS1*, int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef captures(none) %compare, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !111 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !112
  %1 = load i32, ptr addrspace(1) %compare, align 4, !dbg !113
  %2 = cmpxchg ptr addrspace(1) %atomic_address, i32 %1, i32 %0 syncscope("agent") acq_rel monotonic, align 4, !dbg !113
  %3 = extractvalue { i32, i1 } %2, 1, !dbg !113
  br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !113

cmpxchg.store_expected:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, ptr addrspace(1) %compare, align 4, !dbg !113
  br label %cmpxchg.continue, !dbg !113

cmpxchg.continue:
  %conv = zext i1 %3 to i64, !dbg !113
  ret i64 %conv, !dbg !114
}

define internal noundef range(i64 0, 2) i64 @atom_cas_acquire_relaxed_system(int AS1*, int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef captures(none) %compare, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !115 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !116
  %1 = load i32, ptr addrspace(1) %compare, align 4, !dbg !117
  %2 = cmpxchg ptr addrspace(1) %atomic_address, i32 %1, i32 %0 acquire monotonic, align 4, !dbg !117
  %3 = extractvalue { i32, i1 } %2, 1, !dbg !117
  br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !117

cmpxchg.store_expected:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, ptr addrspace(1) %compare, align 4, !dbg !117
  br label %cmpxchg.continue, !dbg !117

cmpxchg.continue:
  %conv = zext i1 %3 to i64, !dbg !117
  ret i64 %conv, !dbg !118
}

define internal noundef range(i64 0, 2) i64 @atom_cas_release_relaxed_system(int AS1*, int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef captures(none) %compare, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !119 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !120
  %1 = load i32, ptr addrspace(1) %compare, align 4, !dbg !121
  %2 = cmpxchg ptr addrspace(1) %atomic_address, i32 %1, i32 %0 release monotonic, align 4, !dbg !121
  %3 = extractvalue { i32, i1 } %2, 1, !dbg !121
  br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !121

cmpxchg.store_expected:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, ptr addrspace(1) %compare, align 4, !dbg !121
  br label %cmpxchg.continue, !dbg !121

cmpxchg.continue:
  %conv = zext i1 %3 to i64, !dbg !121
  ret i64 %conv, !dbg !122
}

define internal noundef range(i64 0, 2) i64 @atom_cas_relaxed_relaxed_system(int AS1*, int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef captures(none) %compare, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !123 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !124
  %1 = load i32, ptr addrspace(1) %compare, align 4, !dbg !125
  %2 = cmpxchg ptr addrspace(1) %atomic_address, i32 %1, i32 %0 monotonic monotonic, align 4, !dbg !125
  %3 = extractvalue { i32, i1 } %2, 1, !dbg !125
  br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !125

cmpxchg.store_expected:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, ptr addrspace(1) %compare, align 4, !dbg !125
  br label %cmpxchg.continue, !dbg !125

cmpxchg.continue:
  %conv = zext i1 %3 to i64, !dbg !125
  ret i64 %conv, !dbg !126
}

define internal noundef range(i64 0, 2) i64 @atom_cas_acqrel_relaxed_system(int AS1*, int AS1*, int AS1*)(ptr addrspace(1) noundef captures(none) %atomic_address, ptr addrspace(1) noundef captures(none) %compare, ptr addrspace(1) noundef readonly captures(none) %value) #0 !dbg !127 {
entry:
  %0 = load i32, ptr addrspace(1) %value, align 4, !dbg !128
  %1 = load i32, ptr addrspace(1) %compare, align 4, !dbg !129
  %2 = cmpxchg ptr addrspace(1) %atomic_address, i32 %1, i32 %0 acq_rel monotonic, align 4, !dbg !129
  %3 = extractvalue { i32, i1 } %2, 1, !dbg !129
  br i1 %3, label %cmpxchg.continue, label %cmpxchg.store_expected, !dbg !129

cmpxchg.store_expected:
  %4 = extractvalue { i32, i1 } %2, 0
  store i32 %4, ptr addrspace(1) %compare, align 4, !dbg !129
  br label %cmpxchg.continue, !dbg !129

cmpxchg.continue:
  %conv = zext i1 %3 to i64, !dbg !129
  ret i64 %conv, !dbg !130
}

declare void @llvm.amdgcn.s.barrier() #2

attributes #0 = { mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) "amdgpu-agpr-alloc"="0" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,+xf32-insts,-tgsplit" "uniform-work-group-size"="false" }
attributes #1 = { convergent mustprogress nofree norecurse nounwind willreturn "amdgpu-agpr-alloc"="0" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,+xf32-insts,-tgsplit" "uniform-work-group-size"="false" }
attributes #2 = { convergent mustprogress nocallback nofree nounwind willreturn }