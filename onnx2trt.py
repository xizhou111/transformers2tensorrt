import tensorrt as trt
from argparse import ArgumentParser

def convert_onnx_to_engine(onnx_filename,
                           engine_filename = None,
                           max_batch_size = 1,
                           max_workspace_size = 1 << 30,
                           fp16_mode = False):
    logger = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(logger) as builder, \
            builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
            trt.OnnxParser(network, logger) as parser, \
            builder.create_builder_config() as config:
        
        config.max_workspace_size = max_workspace_size
        if fp16_mode:
            config.set_flag(trt.BuilderFlag.FP16)
        builder.max_batch_size = max_batch_size

        print("Parsing ONNX file.")
        with open(onnx_filename, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print('Completed parsing of ONNX file')

        print("Building TensorRT engine. This may take a few minutes.")
        engine = builder.build_engine(network, config)
        if engine is None:
            print("Failed to create engine.")
            return None, logger

        if engine_filename:
            with open(engine_filename, 'wb') as f:
                f.write(engine.serialize())
        print("Created engine success! ")

        return engine, logger
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--onnx_path', type=str, required=True, help='Path to the ONNX file')
    parser.add_argument('--output_engine', type=str, required=True, help='Path to save the TensorRT engine')

    args = parser.parse_args()
    onnx_filename = args.onnx_path
    engine_filename = args.output_engine

    convert_onnx_to_engine(onnx_filename, engine_filename, max_batch_size=8, max_workspace_size=1 << 30, fp16_mode=False)