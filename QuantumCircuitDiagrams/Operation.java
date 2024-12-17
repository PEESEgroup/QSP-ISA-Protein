interface Operation<T> {
    T processControlXGate(ControlXGate g);
    T processControlGate(ControlGate g);
    T processUniformControlGate(UniformControlGate g);
    T processRotationGate(RotationGate g);
    T processBigBlock(BigBlock b);
}