<template>
  <b-card class="py-0 titlecard cardbackground frakturfont" body-class="py-0">
    <b-card-title class="my-1"><h2 class="my-0"><b>{{ title }}</b></h2></b-card-title>
  </b-card>
</template>

<script>
export default {
  name: "TitleCard",
  props: {
    title: String
  }
}
</script>

<style scoped>



.titlecard {
  filter: brightness(0.8);
}

</style>
