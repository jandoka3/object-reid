import os
class Arguments():

    datasets = ['Cuhk03_labeled', 'Cuhk03_detected','Market1501', 'VehicleReId', 'VeRi']

    def save_args(self, args):
        if args.dataset not in self.datasets:
            raise Exception("Please select a dataset according to the following options:\n{}".format(self.datasets))

        if args.image_to_track and args.re_rank:
            raise Exception("Re-ranking is not available on image-to-track evaluation.")

        if args.image_to_track:
            args.tracks_dataset = os.path.join('datasets/' + args.dataset, args.dataset + '_tracks.csv')
        args.gallery_dataset = os.path.join('datasets/' + args.dataset, args.dataset + '_test.csv')
        args.query_dataset = os.path.join('datasets/' + args.dataset, args.dataset + '_query.csv')
        args.image_root = os.path.join('datasets', args.dataset + '/Images')
        args.checkpoint = os.path.join('datasets', args.dataset + '/weights/' + args.dataset)
        if args.dataset.startswith('Cuhk03'): args.excluder = 'Cuhk03'
        else: args.excluder = args.dataset
        for key, value in args.__dict__.items():
            self.__dict__[key] = value

    def toString(self):
        for key, value in self.__dict__.items():
            print(key, value)